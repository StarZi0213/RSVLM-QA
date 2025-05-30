import torch
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModel, AutoConfig
import json
import os
import time
from tqdm.autonotebook import tqdm
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B-Instruct"

# 设备配置
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.bfloat16
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float32  # 在MPS上使用float32，因为bfloat16存在兼容性问题
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32  # CPU上使用float32更稳定

print(f"使用设备: {DEVICE} 数据类型: {TORCH_DTYPE}")

# 输入输出文件配置
INPUT_JSONL_FILE = "RSVL-VQA_test.jsonl"  # 你的输入文件
OUTPUT_JSONL_FILE = "internvl3_VQA_test_results.jsonl"  # 结果保存位置

# --- 图像处理常量 ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- 模型和处理器 ---
model = None
tokenizer = None

def build_transform(input_size=448):
    """创建图像转换处理管道"""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """找到最接近原始图像宽高比的目标比例"""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """动态预处理图像，将其分割成多个块"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 计算现有图像的宽高比
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # 找到最接近目标的宽高比
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # 计算目标宽度和高度
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # 调整图像大小
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # 分割图像
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """加载并处理图像"""
    try:
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    except Exception as e:
        tqdm.write(f"加载图像时出错 {image_file}: {e}")
        return None

def load_model_and_tokenizer():
    """加载模型和分词器"""
    global model, tokenizer
    if model is None or tokenizer is None:
        tqdm.write(f"加载模型: {MODEL_ID} 到设备: {DEVICE}")
        try:
            # 根据设备决定是否使用量化
            load_in_8bit = False  # 默认不使用8位量化
            if DEVICE == "cuda":
                # 只在CUDA设备上使用8位量化
                try:
                    import bitsandbytes as bnb
                    load_in_8bit = True
                    tqdm.write("启用8位量化以减少GPU内存使用")
                except ImportError:
                    tqdm.write("未找到bitsandbytes库，将使用全精度模型")
            
            # 确保模型完全加载到指定设备上
            model = AutoModel.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                load_in_8bit=load_in_8bit,  # 仅在CUDA上使用8位量化
                low_cpu_mem_usage=True,
                device_map='auto',  # 自动处理设备映射
                use_flash_attn=False,  # 关闭flash attention避免兼容性问题
                trust_remote_code=True
            ).eval()
            
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, 
                trust_remote_code=True,
                use_fast=False
            )
            tqdm.write("模型和分词器加载成功。")
        except Exception as e:
            tqdm.write(f"加载模型时出错: {e}")
            tqdm.write("请确保有足够的RAM和正确的PyTorch版本。")
            exit(1)

def get_internvl3_answer(image_path, question, retries=2, delay=5, pbar_questions=None, max_new_tokens=1024):
    """
    从InternVL3模型获取给定图像和问题的答案。
    包含基本的重试逻辑。
    pbar_questions是问题内循环的tqdm实例
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        load_model_and_tokenizer()

    try:
        if not os.path.isabs(image_path):
            if pbar_questions: pbar_questions.write(f"警告: 图像路径 {image_path} 似乎是相对路径。假设它是正确的或可访问的。")
            else: tqdm.write(f"警告: 图像路径 {image_path} 似乎是相对路径。假设它是正确的或可访问的。")
        
        pixel_values = load_image(image_path)
        if pixel_values is None:
            return "错误: 无法加载图像"
            
        # 确保像素值与模型在同一设备上并且类型匹配
        pixel_values = pixel_values.to(device=DEVICE, dtype=TORCH_DTYPE)
    except FileNotFoundError:
        if pbar_questions: pbar_questions.write(f"错误: 未找到图像 {image_path}")
        else: tqdm.write(f"错误: 未找到图像 {image_path}")
        return "错误: 未找到图像"
    except Exception as e:
        if pbar_questions: pbar_questions.write(f"打开图像 {image_path} 时出错: {e}")
        else: tqdm.write(f"打开图像 {image_path} 时出错: {e}")
        return f"打开图像时出错: {e}"

    # 为InternVL3格式化问题
    prompt = f"<image>\n{question}"
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=True)

    for attempt in range(retries + 1):
        try:
            # 使用InternVL3的chat方法生成答案
            response = model.chat(tokenizer, pixel_values, prompt, generation_config)
            return response
        except RuntimeError as e:
            error_msg_prefix = f"运行时错误（可能是{DEVICE}内存不足）尝试 {attempt + 1}"
            if "out of memory" in str(e).lower() or "allocated tensor is too large" in str(e):
                if pbar_questions: pbar_questions.write(f"{error_msg_prefix}: {e}")
                else: tqdm.write(f"{error_msg_prefix}: {e}")

                if attempt < retries:
                    if pbar_questions: pbar_questions.write(f"{delay} 秒后重试...")
                    else: tqdm.write(f"{delay} 秒后重试...")
                    time.sleep(delay)
                    if DEVICE == "cuda": torch.cuda.empty_cache()
                    elif DEVICE == "mps": torch.mps.empty_cache()
                else:
                    if pbar_questions: pbar_questions.write("内存不足错误已达到最大重试次数。")
                    else: tqdm.write("内存不足错误已达到最大重试次数。")
                    return f"错误: {DEVICE} 内存不足，重试后仍失败"
            else:
                if pbar_questions: pbar_questions.write(f"尝试 {attempt + 1} 时发生意外的运行时错误: {e}")
                else: tqdm.write(f"尝试 {attempt + 1} 时发生意外的运行时错误: {e}")
                if attempt < retries: time.sleep(delay)
                else: return f"错误: 重试后运行时错误: {e}"
        except Exception as e:
            if pbar_questions: pbar_questions.write(f"InternVL3推理尝试 {attempt + 1} 期间出错: {e}")
            else: tqdm.write(f"InternVL3推理尝试 {attempt + 1} 期间出错: {e}")
            if attempt < retries:
                if pbar_questions: pbar_questions.write(f"{delay} 秒后重试...")
                else: tqdm.write(f"{delay} 秒后重试...")
                time.sleep(delay)
            else: return f"错误: 重试后InternVL3推理失败: {e}"
    return "错误: 推理达到最大重试次数。"

def process_dataset():
    """处理数据集中的所有图像和问题"""
    load_model_and_tokenizer()

    processed_image_ids = set()
    if os.path.exists(OUTPUT_JSONL_FILE):
        try:
            with open(OUTPUT_JSONL_FILE, 'r', encoding='utf-8') as f_out_read:
                for line in f_out_read:
                    try:
                        data = json.loads(line)
                        if "id" in data: processed_image_ids.add(data["id"])
                    except json.JSONDecodeError:
                        tqdm.write(f"警告: 无法解码现有输出文件中的行: {line.strip()}")
            tqdm.write(f"继续处理。在 {OUTPUT_JSONL_FILE} 中找到 {len(processed_image_ids)} 个已处理的图像ID")
        except Exception as e:
            tqdm.write(f"读取现有输出文件 {OUTPUT_JSONL_FILE} 时出错: {e}。从头开始或可能覆盖。")

    # 获取主进度条的总行数
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_count:
            num_total_lines = sum(1 for _ in f_count)
    except FileNotFoundError:
        tqdm.write(f"错误: 未找到输入文件 '{INPUT_JSONL_FILE}'。")
        return
    except Exception as e:
        tqdm.write(f"计算 '{INPUT_JSONL_FILE}' 中的行数时出错: {e}")
        return

    if num_total_lines == 0:
        tqdm.write(f"输入文件 '{INPUT_JSONL_FILE}' 为空。")
        return

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL_FILE, 'a', encoding='utf-8') as f_out:

        # 数据集条目处理的主进度条
        dataset_pbar = tqdm(enumerate(f_in), total=num_total_lines, desc="处理数据集条目")
        for line_number, line in dataset_pbar:
            try:
                entry = json.loads(line)
                image_id = entry["id"]
                original_image_path = entry["image"]

                if image_id in processed_image_ids:
                    # 无需打印，主进度条会跳过
                    continue

                current_image_path = original_image_path  # 假设是绝对路径

                dataset_pbar.set_postfix_str(f"图像ID: {image_id}", refresh=True)  # 更新当前项的描述

                if not os.path.exists(current_image_path):
                    tqdm.write(f"未找到图像文件: {current_image_path} ID {image_id}。跳过行 {line_number + 1}。")
                    error_entry = entry.copy()
                    error_entry["internvl3_processing_error"] = f"未在 {current_image_path} 找到图像文件"
                    error_entry["vqa_pairs_with_internvl3"] = [
                        {**qa_pair, "internvl3_answer": "错误: 未找到图像文件"}
                        for qa_pair in entry.get("vqa_pairs", [])
                    ]
                    f_out.write(json.dumps(error_entry) + "\n")
                    f_out.flush()
                    processed_image_ids.add(image_id)
                    continue

                output_entry = entry.copy()
                output_entry["vqa_pairs_with_internvl3"] = []
                has_errors_for_this_image = False
                
                questions_list = entry.get("vqa_pairs", [])
                # 图像内问题的内部进度条
                questions_pbar = tqdm(enumerate(questions_list), total=len(questions_list), desc=f"图像 {image_id} 问题", leave=False, position=1)
                for i, qa_pair in questions_pbar:
                    question = qa_pair.get("question", "缺少问题")
                    questions_pbar.set_postfix_str(f"问: {question[:30]}...", refresh=True)

                    # 将questions_pbar传递给get_internvl3_answer，以便在需要时使用tqdm.write
                    internvl3_answer = get_internvl3_answer(current_image_path, question, pbar_questions=questions_pbar)

                    new_qa_pair = qa_pair.copy()
                    new_qa_pair["internvl3_answer"] = internvl3_answer
                    output_entry["vqa_pairs_with_internvl3"].append(new_qa_pair)

                    if "错误:" in internvl3_answer:
                        has_errors_for_this_image = True
                
                questions_pbar.close()  # 显式关闭内部进度条

                f_out.write(json.dumps(output_entry) + "\n")
                f_out.flush()
                processed_image_ids.add(image_id)

            except json.JSONDecodeError:
                tqdm.write(f"从行 {line_number + 1} 解码JSON时出错: {line.strip()}")
                f_out.write(json.dumps({"error": "JSONDecodeError", "line_content": line.strip()}) + "\n")
                f_out.flush()
            except KeyError as e:
                tqdm.write(f"处理行 {line_number + 1} 时出现KeyError: {e}。行内容: {line.strip()}")
                f_out.write(json.dumps({"error": f"KeyError: {e}", "line_content": line.strip()}) + "\n")
                f_out.flush()
            except Exception as e:
                tqdm.write(f"处理行 {line_number + 1} 时发生意外错误: {e}, 内容: {line.strip()}")
                f_out.write(json.dumps({"error": "意外处理错误", "exception": str(e), "line_content": line.strip()}) + "\n")
                f_out.flush()
        dataset_pbar.close()  # 显式关闭主进度条

if __name__ == "__main__":
    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"错误: 未找到输入文件 '{INPUT_JSONL_FILE}'。")  # 在tqdm开始前使用print没问题
    else:
        process_dataset()
        print(f"\n处理完成。结果保存到 {OUTPUT_JSONL_FILE}")  # 在tqdm结束后使用print没问题 