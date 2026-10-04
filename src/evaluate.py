"""
Evaluation and Metrics Calculation Module across the 5 Question Types.
"""

import re
import json
from typing import Dict, List, Any

TARGET_CATEGORIES = [
    "object_recognition",
    "color",
    "counting",
    "spatial_relationships",
    "visual_reasoning"
]

def normalize_vqa_answer(ans: str) -> str:
    """
    Standard text normalization for VQA answers:
    - Lowercase
    - Remove punctuation
    - Normalize number words to digits
    """
    if ans is None:
        return ""
    ans = str(ans).lower().strip()
    ans = re.sub(r"[^\w\s]", "", ans)  # Remove punctuation
    ans = re.sub(r"\s+", " ", ans)     # Normalize whitespace

    word_to_num = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
    }
    tokens = ans.split()
    tokens = [word_to_num.get(t, t) for t in tokens]
    return " ".join(tokens)


def evaluate_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates detailed accuracy across the 5 categories and overall.
    Each item in predictions must have:
      - 'category': str
      - 'ground_truth': str
      - 'prediction': str
    """
    stats = {cat: {"correct": 0, "total": 0, "accuracy": 0.0} for cat in TARGET_CATEGORIES}
    total_correct = 0
    total_samples = 0

    for item in predictions:
        cat = item.get("category", "visual_reasoning")
        if cat not in stats:
            stats[cat] = {"correct": 0, "total": 0, "accuracy": 0.0}

        pred_norm = normalize_vqa_answer(item.get("prediction", ""))
        gt_norm = normalize_vqa_answer(item.get("ground_truth", ""))

        is_match = (pred_norm == gt_norm)

        stats[cat]["total"] += 1
        total_samples += 1
        if is_match:
            stats[cat]["correct"] += 1
            total_correct += 1

    # Calculate percentages
    for cat in stats:
        t = stats[cat]["total"]
        c = stats[cat]["correct"]
        stats[cat]["accuracy"] = round((c / t) * 100, 2) if t > 0 else 0.0

    overall_acc = round((total_correct / total_samples) * 100, 2) if total_samples > 0 else 0.0

    return {
        "overall": {
            "correct": total_correct,
            "total": total_samples,
            "accuracy": overall_acc
        },
        "by_category": stats
    }


def generate_markdown_report(metrics: Dict[str, Any], title: str = "Benchmark Evaluation Results") -> str:
    """Formats metrics dictionary into a clean Markdown table."""
    lines = []
    lines.append(f"### {title}\n")
    lines.append("| Question Type (Category) | Correct | Total Samples | Accuracy (%) |")
    lines.append("| :--- | :---: | :---: | :---: |")

    category_display_names = {
        "object_recognition": "Object Recognition",
        "color": "Color",
        "counting": "Counting",
        "spatial_relationships": "Spatial Relationships",
        "visual_reasoning": "Visual Reasoning"
    }

    for cat in TARGET_CATEGORIES:
        stat = metrics["by_category"].get(cat, {"correct": 0, "total": 0, "accuracy": 0.0})
        display = category_display_names.get(cat, cat.replace("_", " ").title())
        lines.append(f"| **{display}** | {stat['correct']:,} | {stat['total']:,} | **{stat['accuracy']:.2f}%** |")

    lines.append("| " + "-"*26 + " | " + "-"*7 + " | " + "-"*13 + " | " + "-"*12 + " |")
    ov = metrics["overall"]
    lines.append(f"| 🏆 **OVERALL TOTAL** | **{ov['correct']:,}** | **{ov['total']:,}** | **{ov['accuracy']:.2f}%** |")
    return "\n".join(lines)


def save_metrics_to_file(metrics: Dict[str, Any], json_path: str, md_path: Optional[str] = None):
    """Save metrics to JSON and Markdown format."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if md_path:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_markdown_report(metrics))
