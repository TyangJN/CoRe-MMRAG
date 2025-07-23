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

    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,  # 使用全部GPU内存
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1}
    )
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)

    logger.info(f"Load {MODEL_NAME} successfully.")

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
                
                max_length = 4500

                _, wiki_content_A = reconstruct_wiki_article(data_wiki_lower, ref_info[0]["wiki_url"], max_length, tokenizer)
                _, wiki_content_B = reconstruct_wiki_article(data_wiki_lower, ref_info[1]["wiki_url"], max_length, tokenizer)
                _, wiki_content_C = reconstruct_wiki_article(data_wiki_lower, ref_info[2]["wiki_url"], max_length, tokenizer)
                _, wiki_content_D = reconstruct_wiki_article(data_wiki_lower, ref_info[3]["wiki_url"], max_length, tokenizer)
                _, wiki_content_E = reconstruct_wiki_article(data_wiki_lower, ref_info[4]["wiki_url"], max_length, tokenizer)



                ref_content_A = f'Reference Image:<image>\nWiki title: {ref_info[0]["wiki_title"]}\nWiki content: {wiki_content_A}'
                # ref_content_B = f'Reference B: Reference wiki image - Image-B:<image>\nWiki title: {ref_info[1]["wiki_title"]}\nWiki content: {wiki_content_B}'
                # ref_content_C = f'Reference C: Reference wiki image - Image-C:<image>\nWiki title: {ref_info[2]["wiki_title"]}\nWiki content: {wiki_content_C}'
                # ref_content_D = f'Reference D: Reference wiki image - Image-D:<image>\nWiki title: {ref_info[3]["wiki_title"]}\nWiki content: {wiki_content_D}'
                # ref_content_E = f'Reference E: Reference wiki image - Image-E:<image>\nWiki title: {ref_info[4]["wiki_title"]}\nWiki content: {wiki_content_E}'

                # question = (
                #     f"Given the question: {question_ori}\n"
                #     f"And Query-Image: <image>\n\n"
                #     f"Hint: The entity in the image is same or similar to {ref_info[0]["wiki_title"]}, {ref_info[1]["wiki_title"]}, {ref_info[2]["wiki_title"]} {ref_info[3]["wiki_title"]} {ref_info[4]["wiki_title"]}\n\n"
                #     f"- please use parametric knowledge and reference material to answer {question_ori} within 5 words"
                # )

                # 不去重
                # question = (
                #     f"Question: {question_ori}\n"
                #     f"Image: <image>\n\n"
                #     f"Note: The image shows an entity that is either one of these or similar type:\n"
                #     f"- {ref_info[0]['wiki_title']}\n"
                #     f"- {ref_info[1]['wiki_title']}\n" 
                #     f"- {ref_info[2]['wiki_title']}\n"
                #     f"- {ref_info[3]['wiki_title']}\n"
                #     f"- {ref_info[4]['wiki_title']}\n\n"
                #     f"- please use parametric knowledge and reference material to answer {question_ori} within 5 words"
                # )

                # 获取唯一的titles
                unique_titles = set(item['wiki_title'] for item in ref_info[:5])

                # 构建titles字符串
                titles_str = "\n".join([f"- {title}" for title in unique_titles])

                # 构建问题
                question = (
                    f"Question: {question_ori}\n"
                    f"Image: <image>\n\n"
                    f"Note: The question is asking about the entity shown in the image:\n"
                    f"- please use parametric knowledge to answer {question_ori} within 5 words"
                )

                img_path = [
                    query_img, 
                    # ref_info[0]['img_path'],
                    # ref_info[1]['img_path'],
                    # ref_info[2]['img_path'],
                    # ref_info[3]['img_path'],
                    # ref_info[4]['img_path'],
                ]

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
                    max_tokens=16,
                    min_tokens=1,
                    temperature=0,
                    best_of=1)
                
                image0 = Image.open(query_img).convert('RGB')
                image1 = Image.open(ref_info[0]['img_path']).convert('RGB')
                image2 = Image.open(ref_info[1]['img_path']).convert('RGB')
                image3 = Image.open(ref_info[2]['img_path']).convert('RGB')
                image4 = Image.open(ref_info[3]['img_path']).convert('RGB')
                image5 = Image.open(ref_info[4]['img_path']).convert('RGB')

                with torch.no_grad():
                    outputs = llm.generate(
                        {
                            "prompt": prompt,
                            "multi_modal_data": {
                                "image": [image0]
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


