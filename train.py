"""
Fine-tuning pipeline for Salesforce/blip-vqa-base optimized for Kaggle GPU T4.
Supports Mixed Precision (FP16), Gradient Scaling, and Checkpointing.
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.dataset import VQADataset, vqa_collate_fn
from src.model import build_blip_model, configure_optimizer_and_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP VQA on Kaggle T4")
    parser.add_argument("--data_dir", type=str, default="./Dataset", help="Path to processed dataset directory")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-vqa-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU (16 works best on 16GB T4)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true", default=True, help="Use Mixed Precision FP16 (recommended for T4)")
    parser.add_argument("--output_dir", type=str, default="./Results/best_model")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"STARTING BLIP VQA TRAINING ON: {device}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"FP16 Mixed Precision: {'Enabled' if args.fp16 else 'Disabled'}")
    print(f"{'='*60}\n")

    # 1. Initialize Model & Processor
    model, processor = build_blip_model(args.model_name, device=device)

    # 2. Prepare DataLoaders
    train_dataset = VQADataset(
        data_dir=args.data_dir,
        split="train",
        processor=processor,
        max_samples=args.max_train_samples
    )
    val_dataset = VQADataset(
        data_dir=args.data_dir,
        split="val",
        processor=processor,
        max_samples=args.max_val_samples
    )

    num_workers = 2 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=vqa_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=vqa_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # 3. Configure Optimizer & LR Scheduler
    total_training_steps = len(train_loader) * args.epochs
    optimizer, scheduler = configure_optimizer_and_scheduler(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        total_training_steps=total_training_steps
    )

    scaler = GradScaler(enabled=(args.fp16 and torch.cuda.is_available()))

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./Results", exist_ok=True)

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=True)
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=(args.fp16 and torch.cuda.is_available())):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}", leave=False):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with autocast(enabled=(args.fp16 and torch.cuda.is_available())):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Save Best Model Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"🔥 New best validation loss ({best_val_loss:.4f})! Saving model to {args.output_dir}...")
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

    # Save training history
    with open("./Results/train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED! Best checkpoint saved at: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
