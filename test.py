"""
Evaluation and Benchmarking Script for BLIP VQA.
Evaluates accuracy across the 5 question types on the Test split:
- Object Recognition
- Color
- Counting
- Spatial Relationships
- Visual Reasoning
"""

import os
import argparse
import json
import torch
from tqdm import tqdm
from PIL import Image

from src.dataset import VQADataset
from src.model import build_blip_model
from src.evaluate import evaluate_predictions, generate_markdown_report, save_metrics_to_file

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BLIP VQA across 5 Question Types")
    parser.add_argument("--data_dir", type=str, default="./Dataset")
    parser.add_argument("--checkpoint", type=str, default="./Results/best_model", help="Path to checkpoint directory or Hugging Face model ID")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Limit number of test samples (None for full test split)")
    parser.add_argument("--output_json", type=str, default="./Results/metrics_summary.json")
    parser.add_argument("--output_md", type=str, default="./Results/evaluation_table.md")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"BENCHMARKING VQA MODEL: {args.checkpoint}")
    print(f"Running on Device: {device}")
    print(f"{'='*60}\n")

    # 1. Load Model & Processor
    model, processor = build_blip_model(args.checkpoint, device=device)
    model.eval()

    # 2. Load Raw Test Dataset (processor=None to get raw PIL image & text for generation)
    test_dataset = VQADataset(
        data_dir=args.data_dir,
        split="test",
        processor=None,
        max_samples=args.max_test_samples
    )

    predictions = []
    print(f"\nRunning inference on {len(test_dataset):,} test samples...")

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating Test Split"):
            item = test_dataset[i]
            image = item["image"]
            question = item["question"]

            inputs = processor(images=image, text=question, return_tensors="pt").to(device)

            # Generate prediction using greedy search
            output_tokens = model.generate(**inputs, max_new_tokens=10)
            pred_text = processor.decode(output_tokens[0], skip_special_tokens=True).strip()

            predictions.append({
                "id": item["id"],
                "image_id": item["image_id"],
                "question": question,
                "ground_truth": item["answer"],
                "prediction": pred_text,
                "category": item["category"]
            })

    # 3. Compute Metrics
    metrics = evaluate_predictions(predictions)

    # 4. Print formatted report
    md_table = generate_markdown_report(metrics, title=f"VQA Performance: {os.path.basename(args.checkpoint)}")
    print("\n" + "="*60)
    print(md_table)
    print("="*60 + "\n")

    # 5. Save results to Results/
    os.makedirs("./Results", exist_ok=True)
    save_metrics_to_file(metrics, args.output_json, args.output_md)

    pred_save_path = "./Results/test_predictions.json"
    with open(pred_save_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ Metrics saved to:     {args.output_json}")
    print(f"✅ Markdown table saved: {args.output_md}")
    print(f"✅ Predictions saved:    {pred_save_path}")

if __name__ == "__main__":
    main()
