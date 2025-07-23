import os
import torch
import json
import logging

from transformers import AutoModel, AutoProcessor, CLIPImageProcessor
import numpy as np
from PIL import Image
from PIL import ImageFile
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 不允许加载截断的图像
Image.MAX_IMAGE_PIXELS = 655530000 * 10  # 设置为所需大小的两倍作为缓冲


class Logger:
    def __init__(self, log_dir, rank, level=logging.INFO, console_output=False):
        # 初始化日志目录
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{timestamp}_rank{rank}.log"

        # 配置logger
        self.logger = logging.getLogger(f"{__name__}_rank{rank}")
        self.logger.setLevel(level)

        # 清除现有处理器
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - '
            f'[Rank {rank}] - '
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


class FeatureExtractor:
    def __init__(self, model_name, model_type, device_id, logger):
        self.logger = logger
        self.device = f'cuda:{device_id}'
        self.model, self.processor = self._load_model(model_name, model_type)
    
    def _load_model(self, model_name, model_type):
        if model_type == "clip":
            # 加载CLIP模型
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            # 删除不需要的组件以避免OOM
            del model.text_projection 
            del model.text_model
            processor = AutoProcessor.from_pretrained(model_name)
            
        elif model_type == "eva-clip":
            # 加载EVA-CLIP模型 
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            # 删除不需要的组件以避免OOM
            del model.text_projection
            del model.text_model
            processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 将模型移至指定设备并包装为DDP
        model = model.to(self.device)
        model = DDP(model, device_ids=[int(self.device.split(':')[1])])
        model.eval()
        
        return model, processor

    def process_batch(self, image_paths, batch_size=512):
        features = []
        valid_paths = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            batch_valid_paths = []

            # 加载图片
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(image)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    self.logger.error(f"Error loading image {img_path}: {e}")
                    continue

            if not batch_images:
                continue

            try:
                # 处理批次
                with torch.no_grad():
                    inputs = self.processor(images=batch_images, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(self.device)
                    batch_features = self.model.module.get_image_features(pixel_values)
                    features.append(batch_features.cpu().numpy())
                    valid_paths.extend(batch_valid_paths)
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")

        return np.concatenate(features) if features else None, valid_paths

def get_rank_data(all_data, rank, world_size):
    """按rank分配数据"""
    per_rank = len(all_data) // world_size
    extra = len(all_data) % world_size
    
    start_idx = rank * per_rank + min(rank, extra)
    end_idx = start_idx + per_rank + (1 if rank < extra else 0)
    
    return all_data[start_idx:end_idx]

def main():
    # 1. 初始化分布式
    local_rank = int(os.environ["LOCAL_RANK"])
    print(local_rank)
    print(os.environ["RANK"])
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(world_size)
    
    # 检查CUDA设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    if local_rank >= torch.cuda.device_count():
        raise ValueError(f"Invalid local_rank {local_rank}, only {torch.cuda.device_count()} GPUs available")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # 2. 初始化日志
    log_dir = '/path/to/your/logs'
    logger = Logger(log_dir)
    logger.info(f"Process {local_rank}/{world_size} started")

    try:
        # 3. 加载数据
        jsonl_path = 'path/to/your/jsonl'

        all_paths = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'image' in data:
                    all_paths.append(data['image'])
                elif 'image_path' in data:
                    all_paths.append(data['image_path'])
                elif 'img_path' in data:
                    all_paths.append(data['img_path'])
                elif 'path' in data:
                    all_paths.append(data['path'])

        # with open(json_path, 'r') as f:
        #     all_paths = json.load(f)

        
        # 4. 数据分片
        rank_paths = get_rank_data(all_paths, local_rank, world_size)
        logger.info(f"Rank {local_rank} processing {len(rank_paths)} images")

        # 5. 特征提取
        extractor = FeatureExtractor(
            model_name="openai/clip-vit-large-patch14", #clip/eva-clip
            device_id=local_rank,
            logger=logger
        )
        
        features, valid_paths = extractor.process_batch(rank_paths)

        # 6. 保存结果
        if features is not None:
            save_dir = Path("path/to/your/jsonl")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            np.save(save_dir / f"features_rank{local_rank}_{timestamp}.npy", features)
            
            with open(save_dir / f"paths_rank{local_rank}_{timestamp}.json", 'w') as f:
                json.dump(valid_paths, f)
                
            logger.info(f"Rank {local_rank} saved results: {len(features)} features")

    except Exception as e:
        logger.error(f"Rank {local_rank} failed: {str(e)}")
        raise
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()