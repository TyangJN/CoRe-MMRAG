import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import json
import logging
from pathlib import Path
from datetime import datetime
from argparse import Namespace
from typing import List, NamedTuple, Optional

from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from vllm.utils import FlexibleArgumentParser


class Logger:
    def __init__(self, log_dir, rank, level=logging.INFO, console_output=False):
        # 初始化日志目录
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"log_{timestamp}_Top_{rank}.log"

        # 配置logger
        self.logger = logging.getLogger(f"{__name__}_Top_{rank}")
        self.logger.setLevel(level)

        # 清除现有处理器
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - '
            f'[Top_{rank}] - '
            '%(levelname)s - '
            '%(message)s'
        )

        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 添加可选的控制台输出
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # 禁止传播到根logger
        self.logger.propagate = False

        # 根logger是层级最顶端的logger
        # root_logger = logging.getLogger()  # 不传名字就是获取根logge

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def exception(self, msg):
        self.logger.exception(msg)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def reconstruct_wiki_article(wiki_kb, wiki_url, max_length, tokenizer):
    """Reconstruct the wiki article from the knowledge entry class."""
    knowledge_entry = wiki_kb[wiki_url]

    title = knowledge_entry['title']
    sub_sections = []  # 用于存储分段内容
    candidate_content = ""  # 当前候选内容
    # max_length = 40000  # 最大长度限制
    count = 0

    for it, section_title in enumerate(knowledge_entry['section_titles']):
        # 跳过不需要的段落
        if (
            "external link" in section_title.lower()
            or "reference" in section_title.lower()
            or "further reading" in section_title.lower()
            or "see also" in section_title.lower()
            or "primary sources" in section_title.lower()
            or "sources" in section_title.lower()
            or "notes" in section_title.lower()
            or "citations" in section_title.lower()
        ):
            continue
        
        # 当前段落内容
        content_added = (
            "\nSection Title: "
            + section_title
            + "\n"
            + knowledge_entry['section_texts'][it]
        )

        tokens = tokenizer.encode(content_added)  

        if count + len(tokens) < max_length:
            candidate_content += content_added
            count = count + len(tokens)

    return title, candidate_content



def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def run_model_on_gpu():
    """
    在指定 GPU 上运行模型，并输出结果和日志。
    """
    # 设置环境变量以使用特定 GPU
    gpu_id = 0

    query_path = 'query.jsonl'

    answer_path = 'answer.jsonl'

    output_dir = ''

    gpu_id = ''
    logger = Logger(output_dir, gpu_id)

    MODEL_NAME = 'your path to qwen2vl'

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,  # 使用全部GPU内存
        max_model_len=32768,
        limit_mm_per_prompt={"image": 6},
        
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

    logger.info(f"Load {MODEL_NAME} successfully.")


    processed_ids = set()
    try:
        with open(answer_path, 'r', encoding='utf-8') as outfile:
            for line in outfile:
                record = json.loads(line.strip())
                if 'id' in record:
                    processed_ids.add(record['id'])
    except FileNotFoundError:
        logger.info("Answer file not found, starting fresh.")

    # 统计总行数
    total_lines = sum(1 for _ in open(query_path, 'r', encoding='utf-8'))


    wiki_kb_path='wiki_100_dict_v4.json'
    with open(wiki_kb_path, 'r') as f_wiki:
        data_wiki = json.load(f_wiki)
    data_wiki_lower = {k.lower(): v for k, v in data_wiki.items()}
    
    with open(query_path, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(tqdm(infile, total=total_lines, desc="Processing", unit="line"), start=1):
            with open(answer_path, 'a', encoding='utf-8') as outfile:

                record = json.loads(line.strip())  # 解析 JSON 数据
                # 跳过已处理的 id
                record_id = record.get('id')
                if record_id in processed_ids:
                    logger.info(f"Skipping already processed record: {record_id}")
                    continue

                ref_info = record['ref_info']
                query_img = record['img_path']
                question_ori = record['question']

                print(record_id)
                
                max_length = 4500

                wiki_content = []
                wiki_title = []
                img_path = [query_img]

                for i in range(5):
                    title_temp, wiki_content_temp = reconstruct_wiki_article(data_wiki_lower, ref_info[i]["wiki_url"], max_length, tokenizer)
                    if title_temp not in wiki_title:
                        wiki_title.append(title_temp)
                        wiki_content.append(wiki_content_temp)
                        img_path.append(ref_info[i]['img_path'])

                ref_content = []
                ref_img_content = []
                
                ref_letter = [] 

                reference_letter = ['A', 'B', 'C', 'D', 'E']
                for j in range(len(wiki_title)):
                    content_temp = f'{reference_letter[j]}: title is {wiki_title[j]}, content is {wiki_content[j]} \n'
                    img_content_temp = f'{reference_letter[j]}: <image>\n '

                    ref_content.append(content_temp)
                    ref_img_content.append(img_content_temp)
                    ref_letter.append(reference_letter[j])

                ref_content_sentence = ''.join(ref_content)
                ref_img_content_sentence = ''.join(ref_img_content)

                letter_choice = '/'.join(ref_letter)

                # question = (
                #     f"Given:\n"
                #     f"- Question: <image>\n{question_ori}\n"
                #     f"Step 1: Direct Answer Check\n"
                #     f"- If the question is directly answerable, provide the answer in the format: [your Answer] and end the inference process.\n"
                #     f"- If not, proceed to Step 2.\n\n"
                #     f"Step 2: Image Relevance Assessment\n"
                #     f"- Based on {ref_img_content_sentence}, judge which image({letter_choice}) is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [Image X] where X is the most similar {letter_choice}.\n\n"
                #     f"Step 3: Content Relevance Assessment\n"
                #     f"- Based on {ref_content_sentence}, judge which content is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [Content Y] where Y is the most similar {letter_choice}.\n\n"
                #     f"Step 4: Conflict Resolution\n"
                #     f"- If the answers from Step 2 and Step 3 differ, determine which one is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [reference Z] where Z is the most similar between X and Y.\n\n"
                #     f"Step 5: Final Answer\n"
                #     f"- Based on the most relevant content identified in Step 4, answer the original question {question_ori} in 5 words."
                #     f"Please output the whole inference process"
                # )

                # question = (
                #     f"Given:\n"
                #     f"- Question: <image>\n{question_ori}\n"
                #     f"Step 1: Direct Answer Check\n"
                #     f"- If the question is directly answerable from the image, provide the answer in the format: [Answer for Step 1] and end the inference process.\n"
                #     f"- If not, proceed to Step 2.\n\n"
                #     f"Step 2: Image Relevance Assessment\n"
                #     f"- Based on {ref_img_content_sentence}, judge which image ({letter_choice}) is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [Answer for Step 2: X], where X is the most similar image from {letter_choice}.\n\n"
                #     f"Step 3: Content Relevance Assessment\n"
                #     f"- Based on {ref_content_sentence}, judge which content is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [Answer for Step 3: X], where X is the most similar option from {letter_choice}.\n\n"
                #     f"Step 4: Conflict Resolution\n"
                #     f"- If the answers from Step 2 and Step 3 differ, determine which one is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [Answer for Step 4: X], where X is the final choice between Step 2 and Step 3.\n\n"
                #     f"Step 5: Final Answer\n"
                #     f"- Based on the most relevant content identified in Step 4, answer the original question {question_ori} in exactly 5 words.\n"
                #     f"- Provide your final answer in the format: [Answer for Step 5].\n\n"
                #     f"Please output the answer for each steps."
                # )


                # question = (
                #     f"Given:\n"
                #     f"- Question: <image>\n{question_ori}\n"
                #     f"Step 1: \n"
                #     f"- If the question is directly answerable from the image, provide the answer in the format: []\n\n"
                #     f"Step 2: \n"
                #     f"- Based on {ref_img_content_sentence}, judge which image ({letter_choice}) is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [X], where X is the most similar image from {letter_choice}.\n\n"
                #     f"Step 3: \n"
                #     f"- Based on {ref_content_sentence}, judge which content is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [X], where X is the most similar option from {letter_choice}.\n\n"
                #     f"Step 4: \n"
                #     f"- If the answers from Step 2 and Step 3 differ, determine which one is more relevant to the Query-Image.\n"
                #     f"- Provide your answer in the format: [X], where X is the final choice between Step 2 and Step 3.\n\n"
                #     f"Step 5: \n"
                #     f"- Based on the most relevant content identified in Step 4, answer the original question {question_ori} in 5 words.\n"
                #     f"- Provide your answer in the format: [].\n\n"
                #     f"Step 6: Final Answer\n"
                #     f"- If the answers from Step 1 and Step 5 differ, determine which one is the final answer, output the final answer in [] within 10 words.\n"
                # )


                # question = (
                #     f"Task: Answer the following question about the image step by step.\n\n"
                #     f"Question: {question_ori}\n"
                #     f"<image>\n\n"
                #     f"Follow these steps:\n\n"
                #     f"1. Direct Answer\n"
                #     f"- Can you answer directly from the image? If yes, write [Step 1 Answer] and stop.\n"
                #     f"- If not, continue to next step.\n\n"
                #     f"2. Image Comparison\n"
                #     f"- Compare query image with options {letter_choice}\n"
                #     f"- Based on {ref_img_content_sentence}\n"
                #     f"- Write [Step 2: X] where X is most similar image\n\n"
                #     f"3. Content Comparison\n"
                #     f"- Compare with content options {letter_choice}\n" 
                #     f"- Based on {ref_content_sentence}\n"
                #     f"- Write [Step 3: X] where X is most similar content\n\n"
                #     f"4. Choose Final Option\n"
                #     f"- If Step 2 and 3 match: use that option\n"
                #     f"- If different: pick more relevant one\n"
                #     f"- Write [Step 4: X] with final choice\n\n"
                #     f"5. Final Answer\n"
                #     f"- Use Step 4's option to answer: {question_ori}\n"
                #     f"- Write [Step 5 answer] within 5 words\n\n"
                #     f"Answer each step directly and clearly."
                # )

                # v4
                # question = (
                #     f"Task: Answer about the image step by step.\n\n"
                #     f"Question: {question_ori}\n"
                #     f"<image>\n\n"
                #     f"Steps:\n\n"
                #     f"1. Check Direct Answer\n"
                #     f"- If answerable from image, write [Answer 1: your_answer_here]\n"
                #     f"- If not answerable, write [Answer 1: None]\n\n"
                #     f"2. Image Comparison\n"
                #     f"- Compare query image with options {letter_choice}\n"
                #     f"- Based on {ref_img_content_sentence}\n"
                #     f"- Write [Answer 2: X] (X = chosen letter)\n\n"
                #     f"3. Content Comparison\n"
                #     f"- Compare query image with options {letter_choice}\n" 
                #     f"- Based on {ref_content_sentence}\n"
                #     f"- Write [Answer 3: X] (X = chosen letter)\n\n"
                #     f"4. Final Choice\n"
                #     f"- Compare Step 2 and 3\n"
                #     f"- Write [Answer 4: X] (X = chosen letter)\n\n"
                #     f"5. Answer Question\n"
                #     f"- Use Step 4's choice\n"
                #     f"- Write [Answer 5: ten_word_answer]\n\n"
                #     f"6. Compare and Conclude\n"
                #     f"- Compare answers from Step 1 and Step 5\n"
                #     f"- Choose most accurate answer\n"
                #     f"- Write [Answer 6: final_answer_in_ten_words]\n\n"
                #     f"Important: Use exact format [Answer N: ...] for all steps"
                # )

                # question = (
                #     f"Task: Answer about the image step by step.\n\n"
                #     f"Question: {question_ori}\n"
                #     f"<image>\n\n"
                #     f"Steps:\n\n"
                #     f"1. Check Direct Answer\n"
                #     f"- If answerable from image, write [Answer 1: ...]\n"
                #     f"- If not answerable, write [Answer 1: None]\n\n"
                #     f"2. Find Best Match\n"
                #     f"- Compare query image with options {letter_choice}\n"
                #     f"- Consider both visual similarity ({ref_img_content_sentence})\n"
                #     f"- AND content relevance ({ref_content_sentence})\n"
                #     f"- Write [Answer 2: X] (X = best matching letter)\n\n"
                #     f"3. Answer Question\n"
                #     f"- Use Step 2's choice\n"
                #     f"- Write [Answer 3: ten_word_answer]\n\n"
                #     f"4. Compare and Conclude\n"
                #     f"- Compare answers from Step 1 and Step 3\n"
                #     f"- Choose most accurate answer\n"
                #     f"- Write [Answer 4: final_answer_in_ten_words]\n\n"
                #     f"Important: Use exact format [Answer N: ...] for all steps"
                # )

                question = (
                    f"Given:\n"
                    f"- Question: <image>\n{question_ori}\n"
                    f"Step 1: Direct Answer Check\n"
                    f"- If the question with image is directly answerable, provide the answer in [] less than 5 words.\n"
                    f"- If not, proceed to Step 2.\n\n"
                    f"Step 2: Relevance Assessment\n"
                    f"- Based on {ref_img_content_sentence} and {ref_content_sentence} answer {question_ori} in less than 5 words.\n\n"
                    f"Step 3: Conflict Resolution\n"
                    f"- If the answers from Step 1 and Step 2 differ, determine which one is the final answer, output the final answer in less than 10 words.\n"
                )


                placeholders = [{"type": "image", "image": path} for path in img_path]

                messages = [{
                    "role": "system",
                    "content": "You are a helpful assistant."
                }, {
                    "role": "user",
                    "content": [
                        *placeholders,
                        {
                            "type": "text",
                            "text": f"{question}"
                        },
                    ],
                }]

                prompt = tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)

                stop_token_ids = None

                # 设置生成参数
                sampling_params = SamplingParams(
                    stop_token_ids=stop_token_ids,
                    max_tokens=512,
                    min_tokens=2,
                    temperature=0,
                    best_of=1)
                
                
                image_list = []
                for path in img_path:
                    image_list.append(Image.open(path).convert('RGB'))
                    
                # image0 = Image.open(query_img).convert('RGB')
                # image1 = Image.open(ref_info[0]['img_path']).convert('RGB')
                # image2 = Image.open(ref_info[1]['img_path']).convert('RGB')
                # image3 = Image.open(ref_info[2]['img_path']).convert('RGB')
                # image4 = Image.open(ref_info[3]['img_path']).convert('RGB')
                # image5 = Image.open(ref_info[4]['img_path']).convert('RGB')

                with torch.no_grad():
                    outputs = llm.generate(
                        {
                            "prompt": prompt,
                            "multi_modal_data": {
                                "image": image_list
                                # "image": [image0, image1, image2, image3, image4, image5]
                                # "image": [image, image]
                            },
                        },
                        sampling_params=sampling_params)
                    torch.cuda.empty_cache()

                    for o in outputs:
                        generated_text = o.outputs[0].text
                        # print(generated_text)
                        record['answer'] = generated_text
                        logger.info(f"{record_id}: {generated_text}")

                    # 写入处理后的记录
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')


run_model_on_gpu()
print("处理完成！")


