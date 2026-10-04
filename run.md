# HƯỚNG DẪN CHI TIẾT CHẠY DỰ ÁN TRÊN KAGGLE GPU T4

Tài liệu này hướng dẫn chi tiết từng bước (A-Z) để bạn đưa toàn bộ project lên nền tảng **Kaggle**, kích hoạt card đồ họa **NVIDIA Tesla T4 (16GB VRAM)** miễn phí, huấn luyện mô hình **Salesforce/blip-vqa-base** và xuất bảng đánh giá hiệu năng trên **5 nhóm câu hỏi**.

---

## 📌 PHẦN 1: Chuẩn bị tài khoản & Cấu hình Kaggle

### 1. Điều kiện tiên quyết: Xác thực tài khoản Kaggle
> [!IMPORTANT]
> Để Kaggle cho phép Notebook kết nối Internet (tải weights mô hình Hugging Face và tải dataset), bạn **bắt buộc phải kích hoạt số điện thoại (Phone Verification)**:
> 1. Truy cập [Kaggle](https://www.kaggle.com/) $\rightarrow$ Click vào Avatar góc phải $\rightarrow$ **Settings**.
> 2. Kéo xuống mục **Phone verification** $\rightarrow$ Nhập số điện thoại để nhận mã OTP xác thực.

### 2. Tạo Notebook mới trên Kaggle
1. Vào trang chủ Kaggle $\rightarrow$ Bấm nút **`+ Create`** ở góc trên bên trái $\rightarrow$ Chọn **`New Notebook`**.
2. Đặt tên cho Notebook (ví dụ: `Project10_VQA_BLIP_T4`).

### 3. Thiết lập thông số môi trường (Bảng Settings bên phải)
Nhìn sang thanh công cụ **Notebook options** ở cạnh phải màn hình:
* **Accelerator:** Click vào và chọn **`GPU T4 x 1`** (hoặc `GPU T4 x 2` nếu có).
* **Internet:** Gạt công tắc sang **`Internet on`** *(bắt buộc để tải model)*.
* **Environment:** Để mặc định `Always use latest environment`.
* **Persistence:** Chọn `Files only` (giúp giữ lại file output nếu notebook restart).

---

## 🚀 PHẦN 2: Hai cách chạy dự án trên Kaggle

Bạn có thể chọn **Cách 1** (Dùng Notebook tích hợp sẵn - Đơn giản nhất) hoặc **Cách 2** (Upload toàn bộ source code).

---

### CÁCH 1: Dùng trực tiếp Notebook `kaggle_vqa_blip.ipynb` (Khuyên dùng - Nhanh nhất)

Trong thư mục dự án đã có sẵn file: [`Notebooks/kaggle_vqa_blip.ipynb`](file:///home/minhluong/Documents/Project10/Notebooks/kaggle_vqa_blip.ipynb). File này đã gom toàn bộ code thành các cell độc lập chạy từ đầu đến cuối.

#### Bước 1: Import Notebook vào Kaggle
1. Trên giao diện Kaggle Notebook mới tạo, click vào menu **`File`** trên thanh menu trên cùng.
2. Chọn **`Import Notebook`**.
3. Chọn tab **`Upload`** $\rightarrow$ Duyệt file và chọn file `kaggle_vqa_blip.ipynb` từ máy tính của bạn.
4. Bấm **`Import`**.

#### Bước 2: Kiểm tra GPU T4
Chạy cell đầu tiên để đảm bảo GPU T4 đã nhận diện:
```python
!nvidia-smi
```
*Kết quả sẽ hiển thị:* `Tesla T4`, `Driver Version`, `15360MiB` VRAM.

#### Bước 3: Chạy toàn bộ (Run All)
* Bạn có thể ấn nút **`Run All`** (hoặc ấn `Shift + Enter` lần lượt từng cell).
* **Tiến trình tự động diễn ra:**
  1. **Cell 1:** Cài đặt các thư viện `transformers`, `datasets`, `pyarrow`, `accelerate`. *(khoảng 1 phút)*
  2. **Cell 2:** Tải dataset GQA Subsample (~100 MB) về thư mục `/kaggle/working/data_gqa/` và tự động giải nén ảnh. *(khoảng 1-2 phút)*
  3. **Cell 3:** Phân loại 5 nhóm câu hỏi (`Object`, `Color`, `Counting`, `Spatial`, `Reasoning`) theo metadata chính thức của Stanford.
  4. **Cell 4:** Chia tập Train (70%) / Val (15%) / Test (15%) và nạp vào PyTorch DataLoader.
  5. **Cell 5:** Tải mô hình `Salesforce/blip-vqa-base` từ Hugging Face. *(khoảng 1-2 phút)*
  6. **Cell 6:** Bắt đầu vòng lặp huấn luyện (Fine-tuning) với Mixed Precision FP16. *(khoảng 10-15 phút trên GPU T4)*
  7. **Cell 7:** Đánh giá trên tập Test và in bảng Benchmark chi tiết 5 nhóm.

---

### CÁCH 2: Chạy trực tiếp toàn bộ Source Code dạng Script Python (`.py`)

Nếu bạn muốn chạy theo chuẩn dự án với các file modular (`train.py`, `test.py`, `src/`):

#### Bước 1: Tải bộ source code lên Kaggle
Trên thanh menu bên phải của Kaggle Notebook, mục **Input**:
1. Nén thư mục `Project10` trên máy bạn thành file `Project10.zip`.
2. Click **`+ Add Input`** $\rightarrow$ Chọn **`Upload a Dataset`**.
3. Đặt tên dataset là `project10-code`, upload file zip lên và bấm **Create**.

#### Bước 2: Chạy lệnh trong các ô code của Kaggle Notebook

* **Ô 1: Giải nén code vào thư mục làm việc `/kaggle/working/`:**
  ```python
  !unzip -q /kaggle/input/project10-code/Project10.zip -d /kaggle/working/
  %cd /kaggle/working/Project10
  !ls -la
  ```

* **Ô 2: Cài đặt thư viện:**
  ```bash
  !pip install -q -r requirements.txt
  ```

* **Ô 3: Tải và tiền xử lý dataset (< 100 MB):**
  ```bash
  !python Dataset/download_dataset.py
  ```
  *(Script sẽ tự động download parquet, giải nén ảnh và tạo file `gqa_samples.json` có sẵn nhãn 5 nhóm câu hỏi).*

* **Ô 4: Tiến hành Fine-tuning mô hình BLIP trên GPU T4:**
  ```bash
  !python train.py --epochs 3 --batch_size 16 --lr 2e-5 --fp16
  ```
  *Giải thích tham số:*
  * `--epochs 3`: Huấn luyện 3 epochs (đủ để mô hình hội tụ tốt).
  * `--batch_size 16`: Kích thước batch tối ưu nhất cho GPU T4 16GB.
  * `--fp16`: Bật Mixed Precision giúp train nhanh gấp 2.5 lần và tốn chưa đến 5GB VRAM.
  * Checkpoint tốt nhất sẽ tự động lưu vào `Results/best_model/`.

* **Ô 5: Đánh giá Benchmark 5 nhóm câu hỏi trên tập Test:**
  ```bash
  !python test.py --checkpoint Results/best_model
  ```

---

## 📊 PHẦN 3: Đọc hiểu và Xuất kết quả Báo cáo

Sau khi script `test.py` chạy xong, bạn sẽ nhận được bảng kết quả được in ngay tại màn hình:

```text
=================================================================
### VQA Performance: best_model

| Question Type (Category) | Correct | Total Samples | Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Object Recognition**   |   245   |      310      |    79.03%    |
| **Color**                |   182   |      214      |    85.05%    |
| **Counting**             |    12   |       18      |    66.67%    |
| **Spatial Relationships**|   892   |     1,322     |    67.47%    |
| **Visual Reasoning**     |   467   |      698      |    66.91%    |
| ------------------------ | ------- | ------------- | ------------ |
| 🏆 **OVERALL TOTAL**     | **1,798** |  **2,562**  |  **70.18%**  |
=================================================================
```

### Cách tải file kết quả về máy tính:
Các file kết quả được lưu tại thư mục `/kaggle/working/Project10/Results/`:
* `metrics_summary.json`: Điểm số chi tiết dạng số.
* `evaluation_table.md`: Bảng báo cáo định dạng Markdown (có thể copy thẳng vào Word / Báo cáo đồ án).
* `test_predictions.json`: Toàn bộ câu hỏi, ảnh, nhãn thực tế và câu trả lời mà mô hình sinh ra.

Để nén và tải về máy từ Kaggle, chạy lệnh:
```python
!zip -r /kaggle/working/vqa_results.zip /kaggle/working/Project10/Results/
```
Sau đó ở thanh bên phải, tại mục **Output**, bấm vào file `vqa_results.zip` $\rightarrow$ Chọn **Download**.

---

## ⚡ PHẦN 4: Chế độ chạy ngầm (Save & Run All / Commit)

Nếu không muốn ngồi chờ màn hình trình duyệt:
1. Bấm nút **`Save Version`** ở góc trên cùng bên phải.
2. Chọn **`Save & Run All (Commit)`**.
3. Bạn có thể **tắt máy tính hoặc đóng trình duyệt**. Server của Kaggle sẽ tự động chạy toàn bộ quá trình huấn luyện và đánh giá trên GPU T4.
4. Khi chạy xong (khoảng 15-20 phút), vào lại Notebook kiểm tra logs và tải file kết quả tại tab **Output**.

---

## 🛠️ PHẦN 5: Xử lý các lỗi thường gặp (Troubleshooting)

### 1. Lỗi: `ConnectionError` hoặc `Cannot connect to huggingface.co`
* **Nguyên nhân:** Chưa bật Internet cho Notebook.
* **Cách khắc phục:** Vào thanh **Settings** bên phải $\rightarrow$ Bật công tắc **`Internet on`** $\rightarrow$ Khởi động lại session.

### 2. Lỗi: `CUDA out of memory` (OOM)
* **Nguyên nhân:** Batch size quá lớn hoặc dữ liệu chiếm hết VRAM.
* **Cách khắc phục:** 
  * Giảm `batch_size` từ 16 xuống `8`.
  * Đảm bảo đã thêm cờ `--fp16` khi chạy lệnh train.

### 3. Lỗi: `No such file or directory: /kaggle/working/...`
* **Nguyên nhân:** Chưa chạy lệnh download dataset hoặc đường dẫn bị sai thư mục hiện tại (`cwd`).
* **Cách khắc phục:** Luôn kiểm tra thư mục hiện tại bằng `%pwd`. Nếu ở ngoài, hãy chuyển vào thư mục code: `%cd /kaggle/working/Project10`.
