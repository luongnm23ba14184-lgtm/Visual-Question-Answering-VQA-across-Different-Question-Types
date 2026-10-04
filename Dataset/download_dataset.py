"""
Dataset Download & Preprocessing Script for Project 10.
Downloads the compact GQA Balanced Subsample (< 100 MB) from Hugging Face
and organizes it into 5 distinct categories:
- object_recognition
- color
- counting
- spatial_relationships
- visual_reasoning
"""

import os
import io
import json
import urllib.request
from typing import Dict, List
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

INSTRUCTIONS_URL = "https://huggingface.co/datasets/lmms-lab/GQA/resolve/main/testdev_balanced_instructions/testdev-00000-of-00001.parquet"
IMAGES_URL = "https://huggingface.co/datasets/lmms-lab/GQA/resolve/main/testdev_balanced_images/testdev-00000-of-00001.parquet"

def map_gqa_to_category(question: str, types_dict: dict) -> str:
    """
    Directly maps GQA official metadata types into the 5 required categories.
    """
    sem = types_dict.get("semantic", "")
    struct = types_dict.get("structural", "")
    det = types_dict.get("detailed", "").lower()
    ql = question.lower().strip()

    # 1. Counting (Quantitative reasoning)
    if "count" in det or ql.startswith("how many") or "number of" in ql:
        return "counting"

    # 2. Color (Visual attribute)
    if "color" in det or "what color" in ql or "which color" in ql:
        return "color"

    # 3. Spatial Relationships (Positional & relational queries)
    if sem == "rel" or any(k in det for k in ["position", "location", "place"]) or any(
        w in ql for w in ["left of", "right of", "behind", "in front of", "next to", "above", "below", "under", "where is", "where are", "on the side", "between"]
    ):
        return "spatial_relationships"

    # 4. Visual Reasoning (Multi-step comparison, logical verification)
    if struct in ["compare", "logical"] or any(k in det for k in ["compare", "logic", "same", "different"]) or ql.startswith("why"):
        return "visual_reasoning"

    # 5. Object Recognition (Object classification, presence, category)
    if sem in ["obj", "cat"] or "object" in det or "category" in det or any(
        ql.startswith(w) for w in ["what is", "what kind of", "what animal", "what vehicle", "is there a", "what are"]
    ):
        return "object_recognition"

    # Default fallback to visual reasoning
    return "visual_reasoning"

def download_file(url: str, output_path: str):
    """Download a file with progress display."""
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return

    print(f"Downloading: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(output_path))
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()

def prepare_dataset(data_dir: str = "./Dataset", extract_images: bool = True):
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    inst_file = os.path.join(data_dir, "instructions.parquet")
    img_file = os.path.join(data_dir, "images.parquet")

    # 1. Download parquet files
    download_file(INSTRUCTIONS_URL, inst_file)
    download_file(IMAGES_URL, img_file)

    # 2. Extract images from parquet to local folder if needed
    print("\nReading image parquet and saving image files...")
    img_table = pq.read_table(img_file)
    # The columns in images.parquet are imageId and image
    img_ids = img_table["imageId"].to_pylist()
    img_bytes = img_table["image"].to_pylist()

    print(f"Extracting {len(img_ids)} unique images to {images_dir}...")
    for img_id, b in tqdm(zip(img_ids, img_bytes), total=len(img_ids)):
        img_out = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_out):
            image = Image.open(io.BytesIO(b["bytes"] if isinstance(b, dict) and "bytes" in b else b))
            image.convert("RGB").save(img_out, format="JPEG", quality=95)

    # 3. Process instructions & categorize
    print("\nProcessing and categorizing QA instructions...")
    inst_table = pq.read_table(inst_file)
    pydict = inst_table.to_pydict()

    processed_samples = []
    category_counts = {
        "object_recognition": 0,
        "color": 0,
        "counting": 0,
        "spatial_relationships": 0,
        "visual_reasoning": 0
    }

    total_rows = len(pydict["id"])
    for i in range(total_rows):
        qid = pydict["id"][i]
        image_id = pydict["imageId"][i]
        question = pydict["question"][i]
        answer = str(pydict["answer"][i]).lower().strip()
        types = pydict["types"][i]

        cat = map_gqa_to_category(question, types)
        category_counts[cat] += 1

        processed_samples.append({
            "id": qid,
            "image_id": image_id,
            "image_path": os.path.join("images", f"{image_id}.jpg"),
            "question": question,
            "answer": answer,
            "category": cat
        })

    # Save processed JSON
    out_json = os.path.join(data_dir, "gqa_samples.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(processed_samples, f, indent=2, ensure_ascii=False)

    info_json = os.path.join(data_dir, "data_info.json")
    with open(info_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(processed_samples),
            "total_images": len(img_ids),
            "category_distribution": category_counts
        }, f, indent=2)

    print("\n" + "="*50)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print(f"Total QA pairs: {len(processed_samples)}")
    print(f"Total Images:   {len(img_ids)}")
    print("Distribution across 5 question types:")
    for cat, count in category_counts.items():
        pct = (count / len(processed_samples)) * 100
        print(f"  - {cat:<22}: {count:>5} ({pct:>5.1f}%)")
    print("="*50)

if __name__ == "__main__":
    prepare_dataset()
