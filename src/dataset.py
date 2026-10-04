"""
PyTorch Dataset and DataLoader module for VQA across Different Question Types.
"""

import os
import json
import random
from typing import Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

class VQADataset(Dataset):
    """
    Dataset class for Visual Question Answering with 5 target categories:
    - object_recognition
    - color
    - counting
    - spatial_relationships
    - visual_reasoning
    """
    def __init__(
        self,
        data_dir: str = "./Dataset",
        split: str = "train",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
        processor = None,
        max_samples: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.processor = processor
        self.split = split

        json_path = os.path.join(data_dir, "gqa_samples.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Dataset not found at {json_path}. Please run 'python Dataset/download_dataset.py' first."
            )

        with open(json_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        # Split by image_id to prevent data leakage between train/val/test
        unique_image_ids = sorted(list(set(s["image_id"] for s in all_samples)))
        random.seed(seed)
        random.shuffle(unique_image_ids)

        n_images = len(unique_image_ids)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        train_imgs = set(unique_image_ids[:n_train])
        val_imgs = set(unique_image_ids[n_train:n_train + n_val])
        test_imgs = set(unique_image_ids[n_train + n_val:])

        if split == "train":
            target_imgs = train_imgs
        elif split == "val":
            target_imgs = val_imgs
        elif split == "test":
            target_imgs = test_imgs
        else:
            raise ValueError(f"Unknown split: {split}. Choose from ['train', 'val', 'test']")

        self.samples = [s for s in all_samples if s["image_id"] in target_imgs]

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(f"[{split.upper()}] Loaded {len(self.samples)} QA pairs across {len(target_imgs)} images.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.samples[idx]
        img_full_path = os.path.join(self.data_dir, item["image_path"])

        try:
            image = Image.open(img_full_path).convert("RGB")
        except Exception as e:
            # Fallback for corrupted images
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        question = item["question"]
        answer = item["answer"]
        category = item["category"]

        if self.processor is not None:
            # BLIP processor tokenizes text and transforms image
            encoding = self.processor(
                images=image,
                text=question,
                padding="max_length",
                max_length=32,
                truncation=True,
                return_tensors="pt"
            )
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # Target answer tokenization
            labels = self.processor.tokenizer(
                answer,
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)

            # Replace padding token id's with -100 so cross entropy ignores them
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            encoding["labels"] = labels
            encoding["category"] = category
            encoding["image_id"] = item["image_id"]
            encoding["id"] = item["id"]
            return encoding

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "category": category,
            "image_id": item["image_id"],
            "id": item["id"]
        }


def vqa_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for training DataLoader."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    categories = [item["category"] for item in batch]
    ids = [item["id"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "categories": categories,
        "ids": ids
    }
