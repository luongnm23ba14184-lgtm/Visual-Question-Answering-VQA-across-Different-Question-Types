# Project 10: Visual Question Answering (VQA) across Different Question Types

> **Hệ thống Hỏi - Đáp Thị Giác (VQA) Đa Dạng Các Nhóm Câu Hỏi trên GPU Kaggle T4**  
> **Model mục tiêu:** `Salesforce/blip-vqa-base`  
> **Nền tảng tối ưu:** Kaggle GPU T4 x 1 / x 2 (16GB VRAM)

---

## 1. Tổng quan Đề tài (Project Overview)

Mục tiêu của đề tài là xây dựng một hệ thống **Visual Question Answering (VQA)** ứng dụng mô hình thị giác - ngôn ngữ (Vision-Language Model), có khả năng tiếp nhận hình ảnh và câu hỏi dạng văn bản để sinh ra câu trả lời chuẩn xác.

Hệ thống được thiết kế và đánh giá chuyên sâu trên **5 dạng câu hỏi cốt lõi** theo đúng yêu cầu đề bài:
1. **Object Recognition (Nhận diện đối tượng):** Xác định sự xuất hiện, tên gọi, chủng loại của các thực thể trong ảnh.
2. **Color (Màu sắc):** Nhận biết và phân biệt chính xác màu sắc của các đối tượng cụ thể.
3. **Counting (Đếm số lượng):** Đếm chính xác số lượng thực thể xuất hiện trong khung hình.
4. **Spatial Relationships (Mối quan hệ không gian):** Hiểu vị trí tương đối giữa các vật thể (*bên trái, bên phải, phía trước, phía sau, bên trên, bên dưới, ở giữa...*).
5. **Visual Reasoning (Suy luận thị giác):** Thực hiện các phép so sánh thuộc tính, liên hệ nhân quả, trạng thái và ngữ cảnh đa bước.

---

## 2. Thông tin & Nguồn Dataset (< 1 GB)

Để đảm bảo hiệu quả huấn luyện trên GPU Kaggle T4, tránh quá tải ổ cứng và có sẵn phân loại chuẩn xác 5 nhóm câu hỏi mà không cần phỏng đoán, dự án sử dụng:

* **Bộ dữ liệu chính:** **GQA Balanced Subsample (Stanford / Hugging Face `lmms-lab/GQA`)**
  * **Link tải chính thức:** [https://huggingface.co/datasets/lmms-lab/GQA](https://huggingface.co/datasets/lmms-lab/GQA)
  * **Dung lượng:** **~65 MB – 150 MB** (hoàn toàn nằm dưới ngưỡng 1 GB).
  * **Đặc điểm:** Ảnh thực tế sắc nét (Visual Genome), câu hỏi được xây dựng từ Scene Graph với metadata chuẩn mực của Stanford:
    * `semantic = 'obj' / 'cat'` $\rightarrow$ **Object Recognition**
    * `semantic = 'attr'` $\rightarrow$ **Color / Attribute**
    * `semantic = 'rel'` $\rightarrow$ **Spatial Relationships**
    * `structural = 'compare' / 'logical'` $\rightarrow$ **Visual Reasoning**
    * Câu hỏi định lượng $\rightarrow$ **Counting**
* **Bộ dữ liệu dự phòng trên Kaggle:** **DAQUAR Processed Dataset**
  * **Link Kaggle:** [https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset](https://www.kaggle.com/datasets/tezansahu/processed-daquar-dataset)
  * **Dung lượng:** **~410 MB** (gồm 1.449 ảnh indoor NYU-Depth V2 và 12.468 cặp QA đã gán sẵn 4 nhóm: Object, Color, Number, Location).

---

## 3. Cấu trúc thư mục Dự án (Project Directory Structure)

Cấu trúc thư mục được đồng bộ chuẩn hóa theo đúng các folder trong workspace:

```text
Project10/
├── Dataset/                   # Dữ liệu tải về và cache parquet/images
│   ├── download_dataset.py    # Script tải tự động dataset < 1GB
│   └── data_info.json         # Thống kê phân bố 5 nhóm câu hỏi
├── Notebooks/                 # Jupyter Notebook chạy trực tiếp trên Kaggle
│   └── kaggle_vqa_blip.ipynb  # Notebook 1-click huấn luyện & đánh giá trên T4
├── Results/                   # Kết quả sau khi chạy
│   ├── best_model/            # Checkpoint weights tốt nhất
│   ├── metrics_summary.json   # Điểm số accuracy chi tiết theo 5 nhóm
│   └── evaluation_table.md    # Bảng kết quả markdown báo cáo
├── src/                       # Mã nguồn mô-đun hóa
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset loader & phân loại 5 categories
│   ├── model.py               # Khởi tạo Salesforce/blip-vqa-base & cấu hình FP16
│   └── evaluate.py            # Hàm tính VQA Accuracy & Category-wise Benchmark
├── train.py                   # Script huấn luyện (Fine-tuning pipeline)
├── test.py                    # Script chạy suy luận & benchmark trên tập Test
├── requirements.txt           # Danh sách thư viện tương thích Kaggle
└── README.md                  # Tài liệu hướng dẫn chi tiết
```

---

## 4. Các Giai đoạn Xây dựng Mô hình (Model Development Phases)

### Giai đoạn 1: Chuẩn bị Dữ liệu & Tiền xử lý (Data Engineering)
* Tải tự động dataset GQA Subsample (< 150 MB).
* Trích xuất thông tin ảnh, câu hỏi, câu trả lời chuẩn hóa và gán nhãn chính xác cho 5 nhóm câu hỏi: `Object`, `Color`, `Counting`, `Spatial`, `Reasoning`.
* Phân chia tập dữ liệu thành **Train (70%)**, **Validation (15%)**, **Test (15%)** theo cơ chế phân tầng (Stratified Splitting) để đảm bảo cả 5 nhóm câu hỏi đều xuất hiện cân đối ở mọi tập.

### Giai đoạn 2: Khởi tạo & Cấu hình Mô hình `Salesforce/blip-vqa-base`
* Load kiến trúc **BLIP (Bootstrapping Language-Image Pre-training)**:
  * Image Encoder: Vision Transformer (ViT-B/16).
  * Text Encoder / Decoder: BERT-based cross-attention decoder.
* Kích hoạt chế độ huấn luyện **Mixed Precision (FP16)** bằng `torch.cuda.amp.autocast`:
  * Tăng tốc độ huấn luyện lên gấp **2.5 lần**.
  * Tiết kiệm bộ nhớ GPU: VRAM sử dụng chỉ **~4.5 GB / 16 GB** trên Kaggle T4.

### Giai đoạn 3: Huấn luyện Tinh chỉnh (Fine-tuning Pipeline)
* Thiết lập siêu tham số tối ưu:
  * Batch size: 16 (hoặc 32 với Gradient Accumulation).
  * Learning rate: $2 \times 10^{-5}$ kết hợp Cosine Annealing Scheduler.
  * Optimizer: AdamW (weight decay = 0.05).
  * Epochs: 3 – 5 epochs.
* Cơ chế lưu trữ: Tự động đánh giá trên tập Validation sau mỗi epoch và lưu lại checkpoint có validation loss thấp nhất vào `Results/best_model/`.

### Giai đoạn 4: Đánh giá Benchmark 5 Nhóm Câu hỏi (Comprehensive Evaluation)
* Thực hiện Greedy Decoding / Beam Search để sinh câu trả lời cho tập Test.
* Áp dụng hàm chuẩn hóa câu trả lời VQA (VQA Normalization: xóa dấu câu, số hóa chữ số, chuyển chữ thường).
* Xuất bảng báo cáo độ chính xác (Accuracy %) tổng thể và chi tiết cho từng nhóm trong 5 nhóm câu hỏi.

---

## 5. Phương pháp Đánh giá (Evaluation Metrics)

Độ chính xác được tính toán độc lập cho từng nhóm câu hỏi:
$$\text{Accuracy}_{\text{Category}} = \frac{\sum_{i=1}^{N_{\text{cat}}} \mathbb{I}(\text{pred}_i = \text{ground\_truth}_i)}{N_{\text{cat}}} \times 100\%$$

Bảng kết quả đánh giá cuối cùng sẽ có định dạng:
| Category | Số mẫu (Samples) | Độ chính xác Baseline (%) | Độ chính xác Sau Fine-tune (%) |
| :--- | :---: | :---: | :---: |
| **Object Recognition** | $N_1$ | -- | -- |
| **Color** | $N_2$ | -- | -- |
| **Counting** | $N_3$ | -- | -- |
| **Spatial Relationships**| $N_4$ | -- | -- |
| **Visual Reasoning** | $N_5$ | -- | -- |
| **OVERALL** | $\sum N_i$ | -- | -- |

---

## 6. Hướng dẫn Chạy trên Kaggle GPU T4 (Kaggle Step-by-Step)

### Cách 1: Chạy bằng file Notebook (Khuyên dùng)
1. Mở Kaggle $\rightarrow$ Chọn **New Notebook**.
2. Tại thanh bên phải (*Settings*):
   * **Accelerator:** Chọn **GPU T4 x 1** (hoặc GPU T4 x 2).
   * **Internet:** Bật **Internet On**.
3. Import file [`Notebooks/kaggle_vqa_blip.ipynb`](file:///home/minhluong/Documents/Project10/Notebooks/kaggle_vqa_blip.ipynb) vào và ấn **Run All**.

### Cách 2: Chạy thông qua dòng lệnh Terminal
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Tải dataset (< 150 MB)
python Dataset/download_dataset.py

# 3. Huấn luyện mô hình BLIP trên GPU T4
python train.py --epochs 3 --batch_size 16 --fp16

# 4. Đánh giá và in bảng kết quả 5 nhóm
python test.py --checkpoint Results/best_model
```
