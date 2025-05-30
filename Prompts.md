### Core Prompts Used in Data Generation and Evaluation

The generation of detailed annotations (tags, relations, VQA pairs, captions) and the subsequent evaluation of model performance heavily rely on carefully engineered prompts provided to Large Language Models (specifically GPT-4.1). Below are the key prompt templates used in our pipeline:

#### 1. Annotation Generation Prompts

These prompts are used to process image descriptions and generate structured data.

* **Tag Extraction:**
    * **Goal:** To extract key remote sensing objects, land cover types, and geographical elements.
    * **Prompt Template:**
        ```
        Extract key remote sensing objects, land cover types, and geographical elements from this aerial/satellite image description. Identify concrete entities (like buildings, roads, water bodies) and land cover types (like forest, urban, agricultural). Return the result as a list of specific tags.
        Image caption: {caption}
        Image summary: {summary}
        Expected output format: {"tags": ["tag1", "tag2", "tag3",...]}
        ```

* **Spatial Relation Extraction:**
    * **Goal:** To identify and extract spatial relationships between objects.
    * **Prompt Template:**
        ```
        Analyze the following text and extract the spatial relationships described in it. Each spatial relationship consists of three parts: object1, directional relationship, and object2. For example, in the sentence ‘The house is next to the lake’, object1 is ‘house’, relation is ‘next to’, and object2 is ‘lake’.
        Text: “{answer}”
        Return the extracted relationships in JSON format as follows:
        <relations> [{"object1": "object1", "relation": "relation", "object2": "object2"}, ...] </relations>
        ```

* **Visual Question Answering (VQA) Pair Generation:**
    * **Goal:** To create diverse VQA pairs based on image content, relations, and tags.
    * **Prompt Template:**
        ```
        Given the following image description, spatial relationships, and object tags, create 9 different types of visual question-answer (VQA) pairs.
        Image Description: {caption},
        Spatial Relationships: {relations},
        Object Tags: {tags}

        Please create 9 different VQA pairs, including:
        1. At least 3 questions about spatial relationships (e.g., 'Where is the highway located?')
        2. At least 2 question about objects in the image (e.g., 'What buildings are visible in the image?')
        3. At least 2 question about overall image features (e.g., 'What type of area is shown?')
        4. At least 2 question about object quantities or proportions (e.g., 'Is there more residential or industrial area?')

        IMPORTANT: Only ask questions that can be answered based on the provided description, relationships, and tags. Do not make up information not present in the input.
        Return in JSON format with question_id, question_type, question, and answer fields:
        ```json
        [
          {
            "question_id": "q1",
            "question_type": "spatial",
            "question": "question text",
            "answer": "answer text"
          },
        ...
        ]
        ```

* **Image Caption, Feature, and Summary Generation (used with `generate_image_captions.py`):**
    * **Goal:** To obtain a comprehensive textual description of an image from GPT-4 Vision.
    * **Prompt Logic (Python f-string from `generate_image_captions.py`):**
        ```python
        # features_str is a string representation of calculated land cover features
        # This is an older version of the prompt, the script generate_image_captions.py has a newer one.
        # prompt = (
        #     f"I'm sending you a satellite/aerial image with these calculated land cover features: {feature_str}.\n\n"
        #     f"Please analyze this image thoroughly and verify if you can see these features. "
        #     f"Don't worry about exact percentages - use descriptive terms like 'large area', 'small portion', etc. "
        #     f"Focus on what's actually visible in the image rather than just repeating the feature list.\n\n"
        #     f"Respond in this exact format:\n"
        #     f"<caption>One comprehensive sentence describing the entire landscape image and its main features</caption>\n"
        #     f"<feature>List the features you can actually see in the image, confirming or contradicting the provided features. Be specific about locations (e.g., 'forest area in the northern section')</feature>\n"
        #     f"<summary>A detailed paragraph summarizing the image content by combining the visual elements and landscape features you identified. THIS SECTION IS REQUIRED AND MUST NOT BE LEFT EMPTY.</summary>"
        # )

        # Current prompt from generate_image_captions.py
        prompt = (
            f"I'm sending you a satellite/aerial image. Please analyze this image thoroughly. Provide directional information from an overall perspective. \
            Regarding the proportion of objects, please Don't give exact percentages.\\You should use descriptive terms like 'large area', 'small portion', etc. \
            Focus on what's actually visible in the image, . \\
            Respond in this exact format: "
            f"<One Sentence caption> describing the entire landscape image and its main features, Total-subtotal structure</One Sentence caption>" # User prompt specifies "<caption>" but script uses "<One Sentence caption>"
            f"<feature>List the features you can actually see in the image, confirming or contradicting the provided features. Be specific about locations (e.g., 'forest area in the northern section')</feature>\n"
            f"<caption>A detailed paragraph summarizing the image content by combining the visual elements and landscape features you identified. THIS SECTION IS REQUIRED AND MUST NOT BE LEFT EMPTY. Total-subtotal structure</caption>" # User prompt specifies "<summary>" but script uses "<caption>" for the detailed summary
        )
        ```
        *Note: The script `generate_image_captions.py` has a slightly different prompt structure in its code than what was initially provided for the "caption" generation type in the user prompt. The version from the script is shown above.*

#### 2. Evaluation Prompts (used with `evaluate_vqa_performance.py`)

These prompts are used by GPT-4.1 to act as an expert evaluator for model-generated answers and captions.

* **For Image Caption Evaluation:**
    * **Goal:** To evaluate the quality of a generated image caption against a reference on a scale of 0-100.
    * **Prompt Template:**
        ```
        You are an expert evaluator for image captioning tasks. Please evaluate the quality of the following caption on a scale of 0-100, where 0 is completely wrong and 100 is perfect.

        Task: {question}
        Reference caption: {ground_truth}
        Generated caption: {model_answer}

        IMPORTANT: Your response MUST FOLLOW this EXACT format without any extra text:
        <score>X</score><reason>Detailed explanation of your score with specific observations</reason>

        Where X is a number between 0 and 100. Do not include any text outside these tags.
        Focus on content accuracy, completeness, detail, and language quality. Be fair but critical in your evaluation.
        ```

* **For Other VQA Task Evaluation:**
    * **Goal:** To judge if a model's answer correctly addresses the question based on a ground truth answer.
    * **Prompt Template:**
        ```
        You are an expert evaluator for visual question answering tasks. Please evaluate if the model's answer properly addresses the question asked, based on the ground truth answer.

        Question: {question}
        Ground truth answer: {ground_truth}
        Model's answer: {model_answer}

        IMPORTANT: Your response MUST FOLLOW this EXACT format without any extra text:
        <judge>Correct</judge><reason>Detailed explanation of your judgment with specific observations</reason>
        OR
        <judge>Wrong</judge><reason>Detailed explanation of your judgment with specific observations</reason>

        Do not include any text outside these tags. Be fair but critical in your evaluation. Consider the semantic meaning rather than exact wording.
        ```
