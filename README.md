# RSVLM-QA: A Benchmark Dataset for Remote Sensing Vision Language Model-based Question Answering

[![Dataset](https://img.shields.io/badge/Dataset-Download-brightgreen)](https://drive.google.com/file/d/1BUAGaZuMFNwNqHxU-NJ-Hi51Ir-yZFwE/view?usp=sharing)
[![Annotations](https://img.shields.io/badge/Annotations-Download-blue)](https://drive.google.com/file/d/1zif3Y95Lfb_0zBy9AX_yTFu1kAzW13tA/view?usp=sharing)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-orange)]() 
[![View Detailed Prompts - Prompts.md](https://img.shields.io/badge/View%20Detailed%20Prompts-Prompts.md-blue)](Prompts.md)
[![VLM Usage Guide - Model Details](https://img.shields.io/badge/VLM%20Usage%20Guide-Model%20Details-green)](models/README.md)
[![View Dataset Demo - HTML Demo](https://img.shields.io/badge/View%20Dataset%20Demo-HTML%20Demo-orange)](DatasetDemo/RSVLM-QA-Demo.html)
[![View Prediction Results - HTML Results](https://img.shields.io/badge/View%20Prediction%20Results-HTML%20Results-purple)](DatasetDemo/VQA-PredictionResults.html)
![RSVLM-QA Dataset Generation Pipeline](assets/pipeline.png)

## Abstract

Visual Question Answering (VQA) in remote sensing (RS) is pivotal for interpreting Earth observation data. However, existing RS VQA datasets are constrained by limitations in annotation richness, question diversity, and the assessment of specific reasoning capabilities. This paper introduces Remote Sensing Vision Language Model Question Answering (RSVLM-QA), a new large-scale, content-rich VQA dataset for the RS domain. RSVLM-QA integrates several well-known RS segmentation and detection datasets, namely WHU, LoveDA, INRIA, and iSAID. We employ an innovative dual-track annotation generation pipeline. Firstly, we leverage Large Language Models (LLMs), specifically GPT-4.1, with meticulously designed prompts to automatically generate a suite of detailed annotations including image captions, spatial relations, and semantic tags, alongside complex caption-based VQA pairs. Secondly, to address the challenging task of object counting in RS imagery, we have developed a specialized automated process that extracts object counts directly from the original segmentation data; GPT-4.1 then formulates natural language answers from these counts, which are paired with preset question templates to create counting QA pairs. RSVLM-QA comprises 13,820 images and 162,373 VQA pairs, featuring extensive annotations and diverse question types. We provide a detailed statistical analysis of the dataset and a comparison with existing RS VQA benchmarks, highlighting the superior depth and breadth of RSVLM-QA's annotations. Furthermore, we conduct benchmark experiments on Six mainstream Vision Language Models (VLMs), demonstrating that RSVLM-QA effectively evaluates and challenges the understanding and reasoning abilities of current VLMs in the RS domain. We believe RSVLM-QA will serve as a pivotal resource for the RS VQA and VLM research communities, poised to catalyze advancements in the field. The dataset, generation code, and benchmark models are publicly available at [https://github.com/StarZi0213/RSVLM-QA](https://github.com/StarZi0213/RSVLM-QA).
## Demos and Results

Explore interactive demonstrations of the dataset and view sample prediction results through the following HTML pages. 
Ensure you have downloaded the repository or have access to these files locally to view them.
Easy way, just download and drop into your browser.

* **RSVLM-QA Interactive Demo:**
    [![View Dataset Demo - HTML Demo](https://img.shields.io/badge/View%20Dataset%20Demo-HTML%20Demo-orange)](DatasetDemo/RSVLM-QA-Demo.html)

* **VQA Prediction Results:**
    [![View Prediction Results - HTML Results](https://img.shields.io/badge/View%20Prediction%20Results-HTML%20Results-purple)](DatasetDemo/VQA-PredictionResults.html)
## Key Features

* **Large-Scale & Content-Rich:** 13,820 images and 162,373 VQA pairs.
* **Diverse Data Sources:** Integrates well-known RS datasets (WHU, LoveDA, INRIA, iSAID).
* **Innovative Dual-Track Generation:**
    * **Track 1 (LLM-driven):** GPT-4.1 generates detailed image captions, spatial relations, semantic tags, and complex caption-based VQA pairs.
    * **Track 2 (Automated Counting):** Specialized process extracts object counts from segmentation data, with natural language answers for counting QA pairs.
* **Extensive Annotations:** Superior depth and breadth compared to existing RS VQA benchmarks.
* **Challenging Benchmarks:** Designed to effectively evaluate and push the boundaries of VLMs in the RS domain.
* **Open Source:** Dataset, generation code, and model benchmarking scripts are publicly available.

## Dataset Access

* **Full Dataset (Images):** [Download from Google Drive](https://drive.google.com/file/d/1BUAGaZuMFNwNqHxU-NJ-Hi51Ir-yZFwE/view?usp=sharing)
* **Annotation Files (JSONL):** [Download from Google Drive](https://drive.google.com/file/d/1zif3Y95Lfb_0zBy9AX_yTFu1kAzW13tA/view?usp=sharing)

  
### Core Prompts Used in Data Generation and Evaluation

The generation of detailed annotations (tags, relations, VQA pairs, captions) and the subsequent evaluation of model performance heavily rely on carefully engineered prompts provided to Large Language Models (specifically GPT-4.1).

For a comprehensive list and detailed explanations of all prompts used in our pipeline, please refer to the dedicated `Prompts.md` file:

[![View Detailed Prompts - Prompts.md](https://img.shields.io/badge/View%20Detailed%20Prompts-Prompts.md-blue)](Prompts.md)

The `Prompts.md` file includes the specific templates for:
* Tag Extraction
* Spatial Relation Extraction
* Visual Question Answering (VQA) Pair Generation
* Image Caption, Feature, and Summary Generation (for GPT-4 Vision)
* Evaluation of Image Captions
* Evaluation of VQA Tasks (Correct/Wrong judgment)

  
## Dataset Generation Pipeline

The RSVLM-QA dataset was created using a sophisticated pipeline involving both automated segmentation data processing and advanced Large Language Model capabilities. The core scripts for this pipeline are provided in this repository:

1.  **Image Caption and Feature Generation (`generate_image_captions.py`):**
    * Uses GPT-4 Vision (gpt-4.1) to analyze remote sensing images.
    * Generates descriptive captions, identifies key visual features, and provides a detailed summary for each image.
    * Handles API rate limits, retries, and checkpointing for robust processing.
    * Converts TIFF images to JPEG for compatibility with the API.
    * Outputs results in JSONL format, including the image path, extracted features, and GPT-4's textual analysis (caption, feature analysis, summary).

2.  **Counting-based VQA Pair Generation (`generate_count_vqa_pairs.py`):**
    * Processes records containing segmentation data (`Seg` field).
    * Extracts object counts for different categories from the segmentation information.
    * Generates various types of counting-related questions:
        * Direct count questions (e.g., "How many cars are there?")
        * Presence questions (e.g., "Are there any buildings?")
        * Comparative count questions (e.g., "Are there more trees or roads?")
        * Total object count questions.
    * Formulates natural language answers based on the extracted counts.
    * Includes checkpointing and robust error handling.
    * Outputs JSONL files with records updated to include `count_vqa_pairs`.

3.  **VQA Performance Evaluation (`evaluate_vqa_performance.py`):**
    * Evaluates the answers provided by various Visual Language Models (VLMs) against ground truth answers.
    * Uses GPT-4.1 as an expert evaluator.
    * Provides different evaluation prompts for image captioning tasks (score-based) and other VQA tasks (Correct/Wrong judgment).
    * Manages API rate limits, multithreading, checkpointing, and detailed statistics collection.
    * Input: JSONL file with model answers (e.g., from VLM inference scripts).
    * Output: JSONL file with evaluation results (judgment and reason) appended.

## Model Benchmarking

We have benchmarked several state-of-the-art Vision Language Models (VLMs) on the RSVLM-QA dataset. The inference scripts and usage instructions for these models are provided in the `models/` directory.

Please refer to the **[Visual Language Models Usage Guide (models/README.md)](models/README.md)** for detailed setup, configuration, and execution instructions for each of the following models:

* Qwen2.5-VL
* Gemma 3
* BLIP2
* InternVL3
* LLaVA
* OVIS2

The `models/README.md` covers:
* Common setup and installation.
* Model-specific dependencies and configurations.
* Instructions on how to run inference.
* Performance optimization tips (memory and speed).
* Details on batch processing and data format specifications.

## General Installation (for Data Generation & Evaluation Scripts)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/StarZi0213/RSVLM-QA.git
    cd RSVLM-QA
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install common dependencies:**
    The primary scripts (`generate_image_captions.py`, `generate_count_vqa_pairs.py`, `evaluate_vqa_performance.py`) have dependencies like `requests`, `Pillow`, `tqdm`, `backoff`.
    ```bash
    pip install requests Pillow tqdm backoff
    ```
    *Note: For specific VLM model inference, refer to the installation instructions in `models/README.md`.*

4.  **API Keys:**
    * The data generation and evaluation scripts require an OpenAI API key for GPT-4.1. You can set it as an environment variable `OPENAI_API_KEY` or pass it via the `--api-key` argument when running the scripts.

## Usage Workflow Example

1.  **Download Dataset:** Obtain the RSVLM-QA images and initial annotation files from the links above.
2.  **Data Generation (Optional - if you want to regenerate or customize):**
    * Use `generate_image_captions.py` to create/update image descriptions.
    * Use `generate_count_vqa_pairs.py` to generate VQA pairs based on object counts from segmentation data.
3.  **Run VLM Inference:**
    * Follow instructions in `models/README.md` to run inference with one of the supported VLMs (e.g., Qwen2.5-VL, BLIP2) on the RSVLM-QA dataset. This will produce a JSONL file with the model's answers.
4.  **Evaluate VLM Performance:**
    * Use `evaluate_vqa_performance.py` with the output from the VLM inference to get GPT-4.1 based evaluations.
    * Example:
        ```bash
        python evaluate_vqa_performance.py \
          --input path/to/your/model_VQA_test_results.jsonl \
          --output path/to/your/model_VQA_test_results_GPT4.1Evaled.jsonl \
          --api-key YOUR_OPENAI_API_KEY \
          --threads 10
        ```
        
## Citation

If you use the RSVLM-QA dataset or the code from this repository in your research, please cite our work (details will be updated upon publication):

```bibtex
@misc{rsvlmqa2025,
  title={RSVLM-QA: A Benchmark Dataset for Remote Sensing Vision Language Model-based Question Answering},
  author={Zi, Xing and Xiao, Jinghao and Shi, Yunxiao and Tao, Xian and Li, Jun and Braytee, Ali and Prasad, Mukesh},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{[https://github.com/StarZi0213/RSVLM-QA](https://github.com/StarZi0213/RSVLM-QA)}}
}
