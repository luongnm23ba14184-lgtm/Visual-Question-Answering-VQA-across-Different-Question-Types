"""
Model management module for Salesforce/blip-vqa-base on Kaggle GPU T4.
"""

import os
from typing import Tuple, Optional
import torch
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    get_cosine_schedule_with_warmup
)

MODEL_NAME = "Salesforce/blip-vqa-base"

def build_blip_model(
    model_name_or_path: str = MODEL_NAME,
    device: Optional[torch.device] = None
) -> Tuple[BlipForQuestionAnswering, BlipProcessor]:
    """
    Initializes and returns the BLIP VQA model and processor.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading BLIP VQA model from: {model_name_or_path}...")
    processor = BlipProcessor.from_pretrained(model_name_or_path)
    model = BlipForQuestionAnswering.from_pretrained(model_name_or_path)
    model = model.to(device)

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable Parameters: {trainable_params:,}")
    print(f"Model placed on: {device}")

    return model, processor


def configure_optimizer_and_scheduler(
    model: BlipForQuestionAnswering,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.05,
    total_training_steps: int = 1000,
    warmup_ratio: float = 0.1
):
    """
    Sets up AdamW optimizer with weight decay and Cosine Annealing learning rate schedule.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    warmup_steps = int(total_training_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    return optimizer, scheduler
