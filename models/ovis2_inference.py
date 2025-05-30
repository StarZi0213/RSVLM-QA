import os
import json
import torch
from PIL import Image
import requests
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from io import BytesIO
import argparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

# --- 配置 ---
MODEL_ID = "AIDC-AI/Ovis2-8B"

# 设备选择逻辑
if torch.backends.mps.is_available():
    DEVICE = "mps"
    TORCH_DTYPE = torch.float16  # MPS上float16通常更快且内存占用更少
elif torch.cuda.is_available():
    DEVICE = "cuda"  # 对于NVIDIA GPU
    TORCH_DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32  # CPU上float32更稳定

print(f"使用设备: {DEVICE} 数据类型: {TORCH_DTYPE}")

INPUT_JSONL_FILE = "json/RSVL-VQA_test.jsonl"  # 输入文件
OUTPUT_JSONL_FILE = "json/Test-Results/ovis2_VQA_test_results.jsonl"  # 结果保存位置
CHECKPOINT_FILE = "ovis2_inference_checkpoint.json"  # 检查点文件

# --- 加载模型和处理器 ---
# 初始化全局变量
model = None
text_tokenizer = None
visual_tokenizer = None

def load_model_and_tokenizers():
    """加载Ovis2-8B模型和分词器"""
    global model, text_tokenizer, visual_tokenizer
    if model is None or text_tokenizer is None or visual_tokenizer is None:
        tqdm.write(f"正在加载模型: {MODEL_ID} 到设备: {DEVICE}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                multimodal_max_length=32768,
                trust_remote_code=True
            )
            
            if DEVICE == "cuda":
                model = model.cuda()
            elif DEVICE == "mps":
                model = model.to("mps")
            
            text_tokenizer = model.get_text_tokenizer()
            visual_tokenizer = model.get_visual_tokenizer()
            
            tqdm.write("模型和分词器加载成功。")
        except Exception as e:
            tqdm.write(f"加载模型时出错: {e}")
            tqdm.write("请确保有足够的RAM和正确的PyTorch版本。")
            exit()

def get_ovis2_answer(image_path, question, retries=2, delay=5, pbar_questions=None, max_new_tokens=1024, max_partition=9):
    """
    使用Ovis2-8B模型为给定图像和问题生成答案。
    包含基本重试逻辑。
    pbar_questions是内部循环(问题)的tqdm实例
    """
    global model, text_tokenizer, visual_tokenizer

    if model is None or text_tokenizer is None or visual_tokenizer is None:
        load_model_and_tokenizers()

    try:
        if not os.path.isabs(image_path):
            if pbar_questions: pbar_questions.write(f"警告: 图像路径 {image_path} 似乎是相对路径。假设它是正确的或可访问的。")
            else: tqdm.write(f"警告: 图像路径 {image_path} 似乎是相对路径。假设它是正确的或可访问的。")
            current_image_path = image_path
        else:
            current_image_path = image_path

        # 打开图像
        try:
            image = Image.open(current_image_path).convert('RGB')
            images = [image]
        except FileNotFoundError:
            if pbar_questions: pbar_questions.write(f"错误: 在 {current_image_path} 未找到图像")
            else: tqdm.write(f"错误: 在 {current_image_path} 未找到图像")
            return "错误: 未找到图像"
        except Exception as e:
            if pbar_questions: pbar_questions.write(f"打开图像 {current_image_path} 时出错: {e}")
            else: tqdm.write(f"打开图像 {current_image_path} 时出错: {e}")
            return f"打开图像时出错: {e}"
    
        # 准备查询文本
        query = f"<image>\n{question}"
        
    except Exception as e:
        if pbar_questions: pbar_questions.write(f"准备输入时出错: {e}")
        else: tqdm.write(f"准备输入时出错: {e}")
        return f"准备输入时出错: {e}"

    for attempt in range(retries + 1):
        try:
            # 为潜在的长生成，最好不要在这个紧密循环中打印太多内容
            # 问题进度条将显示活动。
            
            # 格式化会话并准备输入
            prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
            attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
            
            pixel_values = [pixel_values]
            
            # 生成输出
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                    repetition_penalty=None,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=text_tokenizer.pad_token_id,
                    use_cache=True
                )
                output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
                answer = text_tokenizer.decode(output_ids, skip_special_tokens=True)
                
                # 清除输入提示
                answer = answer.replace(query, "").strip()
                
                return answer
                
        except RuntimeError as e:
            error_msg_prefix = f"运行时错误 (可能在 {DEVICE} 上内存不足) 尝试 {attempt + 1}"
            if "CUDA out of memory" in str(e) or "MPS backend out of memory" in str(e) or "allocated tensor is too large" in str(e):
                if pbar_questions: pbar_questions.write(f"{error_msg_prefix}: {e}")
                else: tqdm.write(f"{error_msg_prefix}: {e}")

                if attempt < retries:
                    if pbar_questions: pbar_questions.write(f"{delay} 秒后重试...")
                    else: tqdm.write(f"{delay} 秒后重试...")
                    time.sleep(delay)
                    if DEVICE == "mps": torch.mps.empty_cache()
                    elif DEVICE == "cuda": torch.cuda.empty_cache()
                else:
                    if pbar_questions: pbar_questions.write("已达到OOM错误的最大重试次数。")
                    else: tqdm.write("已达到OOM错误的最大重试次数。")
                    return f"错误: {DEVICE} 重试后内存不足"
            else:
                if pbar_questions: pbar_questions.write(f"尝试 {attempt + 1} 时发生意外的运行时错误: {e}")
                else: tqdm.write(f"尝试 {attempt + 1} 时发生意外的运行时错误: {e}")
                if attempt < retries: time.sleep(delay)
                else: return f"错误: 重试后运行时错误: {e}"
        except Exception as e:
            if pbar_questions: pbar_questions.write(f"尝试 {attempt + 1} 时Ovis2推理期间出错: {e}")
            else: tqdm.write(f"尝试 {attempt + 1} 时Ovis2推理期间出错: {e}")
            if attempt < retries:
                if pbar_questions: pbar_questions.write(f"{delay} 秒后重试...")
                else: tqdm.write(f"{delay} 秒后重试...")
                time.sleep(delay)
            else: return f"错误: 重试后Ovis2推理失败: {e}"
    
    return "错误: 推理达到最大重试次数。"

def process_dataset():
    """处理整个数据集"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSONL_FILE), exist_ok=True)
    
    load_model_and_tokenizers()

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

    # 检查是否存在检查点，并加载已处理的ID
    if os.path.exists(CHECKPOINT_FILE) and len(processed_image_ids) == 0:
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint_data = json.load(f)
                processed_ids_from_checkpoint = set(checkpoint_data.get('processed_ids', []))
                processed_image_ids.update(processed_ids_from_checkpoint)
            tqdm.write(f"已从检查点恢复: {len(processed_ids_from_checkpoint)} 条记录已处理")
        except Exception as e:
            tqdm.write(f"读取检查点文件时出错: {e}")

    # 获取总行数用于主进度条
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_count:
            num_total_lines = sum(1 for _ in f_count)
    except FileNotFoundError:
        tqdm.write(f"错误: 未找到用于计数行的输入文件 '{INPUT_JSONL_FILE}'。")
        return
    except Exception as e:
        tqdm.write(f"计数 '{INPUT_JSONL_FILE}' 中的行时出错: {e}")
        return

    if num_total_lines == 0:
        tqdm.write(f"输入文件 '{INPUT_JSONL_FILE}' 为空。")
        return

    # 记录开始时间
    start_time = time.time()

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_JSONL_FILE, 'a', encoding='utf-8') as f_out:

        # 数据集处理的主进度条
        dataset_pbar = tqdm(enumerate(f_in), total=num_total_lines, desc="处理数据集条目")
        for line_number, line in dataset_pbar:
            try:
                entry = json.loads(line)
                image_id = entry["id"]
                original_image_path = entry["image"]

                if image_id in processed_image_ids:
                    # 不需要在这里打印，主进度条将直接跳过
                    continue

                current_image_path = original_image_path  # 假设是绝对路径

                dataset_pbar.set_postfix_str(f"图像ID: {image_id}", refresh=True)  # 更新当前项的描述

                if not os.path.exists(current_image_path):
                    tqdm.write(f"未找到图像文件: ID {image_id} 的 {current_image_path}。跳过第 {line_number + 1} 行。")
                    error_entry = entry.copy()
                    error_entry["ovis2_processing_error"] = f"在 {current_image_path} 未找到图像文件"
                    error_entry["vqa_pairs_with_ovis2"] = [
                        {**qa_pair, "ovis2_answer": "错误: 未找到图像文件"}
                        for qa_pair in entry.get("vqa_pairs", [])
                    ]
                    f_out.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                    f_out.flush()
                    processed_image_ids.add(image_id)
                    continue

                output_entry = entry.copy()
                output_entry["vqa_pairs_with_ovis2"] = []
                has_errors_for_this_image = False
                
                questions_list = entry.get("vqa_pairs", [])
                # 图像内问题的内部进度条
                questions_pbar = tqdm(enumerate(questions_list), total=len(questions_list), desc=f"图像 {image_id} 问题", leave=False, position=1)
                for i, qa_pair in questions_pbar:
                    question = qa_pair.get("question", "缺失问题")
                    questions_pbar.set_postfix_str(f"问题: {question[:30]}...", refresh=True)

                    # 将questions_pbar传递给get_ovis2_answer，以便在需要时使用tqdm.write
                    ovis2_answer = get_ovis2_answer(current_image_path, question, pbar_questions=questions_pbar)

                    new_qa_pair = qa_pair.copy()
                    new_qa_pair["ovis2_answer"] = ovis2_answer
                    output_entry["vqa_pairs_with_ovis2"].append(new_qa_pair)

                    if "错误:" in ovis2_answer:
                        has_errors_for_this_image = True
                
                questions_pbar.close()  # 显式关闭内部进度条

                # 写入完整的处理结果
                f_out.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
                f_out.flush()
                processed_image_ids.add(image_id)
                
                # 处理完一个项目后更新检查点
                with open(CHECKPOINT_FILE, 'w') as f:
                    json.dump({
                        "processed_ids": list(processed_image_ids),
                        "timestamp": time.time()
                    }, f)

            except json.JSONDecodeError:
                tqdm.write(f"解码第 {line_number + 1} 行的JSON时出错: {line.strip()}")
                f_out.write(json.dumps({"error": "JSONDecodeError", "line_content": line.strip()}, ensure_ascii=False) + "\n")
                f_out.flush()
            except KeyError as e:
                tqdm.write(f"处理第 {line_number + 1} 行时键错误: {e}。行内容: {line.strip()}")
                f_out.write(json.dumps({"error": f"KeyError: {e}", "line_content": line.strip()}, ensure_ascii=False) + "\n")
                f_out.flush()
            except Exception as e:
                tqdm.write(f"处理第 {line_number + 1} 行时发生意外错误: {e}, 内容: {line.strip()}")
                f_out.write(json.dumps({"error": "意外处理错误", "exception": str(e), "line_content": line.strip()}, ensure_ascii=False) + "\n")
                f_out.flush()
        dataset_pbar.close()  # 显式关闭主进度条
    
    # 处理完成后，删除检查点文件
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
        except Exception as e:
            tqdm.write(f"删除检查点文件时出错: {e}")
    
    # 计算处理时间
    elapsed_time = time.time() - start_time
    tqdm.write(f"处理完成! 总计处理 {len(processed_image_ids)} 条记录")
    tqdm.write(f"处理时间: {elapsed_time:.2f} 秒")
    if len(processed_image_ids) > 0:
        tqdm.write(f"每条记录平均处理时间: {elapsed_time/len(processed_image_ids):.2f} 秒")
    tqdm.write(f"结果已保存到: {OUTPUT_JSONL_FILE}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='使用Ovis2-8B模型进行视觉问答推理')
    parser.add_argument('--input', type=str, default=INPUT_JSONL_FILE,
                        help='输入JSONL文件路径')
    parser.add_argument('--output', type=str, default=OUTPUT_JSONL_FILE,
                        help='输出JSONL文件路径')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='生成的最大新token数')
    parser.add_argument('--max_partition', type=int, default=9,
                        help='图像分区的最大数量')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # 更新全局配置
    INPUT_JSONL_FILE = args.input
    OUTPUT_JSONL_FILE = args.output
    
    if not os.path.exists(INPUT_JSONL_FILE):
        print(f"错误: 未找到输入文件 '{INPUT_JSONL_FILE}'。")
    else:
        try:
            process_dataset()
            print(f"\n处理完成。结果已保存到 {OUTPUT_JSONL_FILE}")
        except KeyboardInterrupt:
            print("\n程序被中断，正在保存检查点...")
            print("您可以使用相同的命令重新启动程序，它将从中断点继续。") 