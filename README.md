# 🔬 XLA Cell & Particle Counter

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-Copyright%20Protected-red)

**XLA Cell Counter** là ứng dụng xử lý ảnh số (Digital Image Processing) hỗ trợ tự động phân đoạn, tách biệt và đếm số lượng tế bào hoặc hạt vi mô từ hình ảnh hiển vi và ảnh thực tế. Dự án tập trung giải quyết bài toán các đối tượng có kích thước không đồng đều, nhiễu nền phức tạp và dính chùm (overlapping/touching objects).

---

## ✨ Tính năng nổi bật

* **Tiền xử lý ảnh đa kênh (Image Preprocessing):** Khử nhiễu, cân bằng sáng và tăng cường độ tương phản cho cả ảnh huỳnh quang (Fluorescence), ảnh nhuộm mô học (Stained) và ảnh vật thể thực tế.
* **Phân đoạn nâng cao (Advanced Segmentation):** Kết hợp phân ngưỡng thích nghi (Adaptive/Otsu Thresholding), các phép toán hình thái học (Morphological Operations), biến đổi khoảng cách (**Distance Transform**) và thuật toán **Watershed** để tách chính xác các tế bào dính nhau.
* **Hậu xử lý & Lọc nhiễu:** Lọc đối tượng theo diện tích, độ tròn và hình thái nhằm loại bỏ tạp chất hoặc nhiễu hạt.
* **Trực quan hóa kết quả (Visualization):** Khoanh vùng đường viền (Contours), đánh số thứ tự từng đối tượng và thống kê tổng số lượng trực tiếp trên giao diện.

---

## 📂 Cấu trúc dự án

```text
xla_cell_counter/
├── app.py                  # Giao diện người dùng (GUI) và điều phối luồng chạy chính
├── counter_core.py         # Lõi thuật toán xử lý ảnh, phân đoạn và đếm đối tượng
├── count_helper.py         # Các hàm tiện ích hỗ trợ tiền xử lý, tính toán và vẽ kết quả
├── dataset/                # Tập dữ liệu ảnh mẫu dùng để kiểm thử
│   ├── cell_01_fluo.png    # Ảnh tế bào huỳnh quang (Fluorescence)
│   ├── cell_03_stain.png   # Ảnh tế bào nhuộm màu (Stained)
│   ├── an-dau-phong-2-5627.png # Ảnh hạt thực tế kiểm thử mở rộng
│   └── ...
├── .gitignore              # Cấu hình loại trừ các file tạm và môi trường ảo
└── README.md               # Tài liệu hướng dẫn và thông tin bản quyền dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Yêu cầu hệ thống
* Python `3.8` trở lên.
* Các thư viện xử lý ảnh và giao diện: `opencv-python`, `numpy`, `matplotlib` (hoặc `PySide6`/`tkinter` tùy cấu hình GUI).

### 2. Cài đặt

Clone kho lưu trữ về máy cá nhân:
```bash
git clone [https://github.com/ShadowPacket07/xla-cell-counter.git](https://github.com/ShadowPacket07/xla-cell-counter.git)
cd xla-cell-counter
```

Khởi tạo môi trường ảo và cài đặt thư viện:
```bash
python -m venv venv
# Kích hoạt môi trường ảo trên Windows:
venv\Scripts\activate
# Hoặc trên macOS/Linux:
source venv/bin/activate

# Cài đặt các thư viện cần thiết:
pip install opencv-python numpy matplotlib
```

### 3. Khởi chạy ứng dụng
```bash
python app.py
```

---

## 🧪 Tập dữ liệu kiểm thử (Dataset)

Thư mục `dataset/` đi kèm dự án bao gồm 3 nhóm ảnh chính để đánh giá độ ổn định của thuật toán:
1. **Fluorescence Microscopy (`cell_*_fluo.png`):** Ảnh tế bào phát quang trên nền tối, kiểm thử khả năng tách biên khi mật độ tế bào dày.
2. **Stained Cells (`cell_*_stain.png`):** Ảnh tế bào nhuộm mô học với độ tương phản nền phức tạp.
3. **Real-world Objects (`an-dau-phong-*.png`):** Ảnh vật thể thực tế dùng để kiểm chứng tính tổng quát của thuật toán phân đoạn.

---

## ⚖️ Tuyên bố Bản quyền & Liêm chính Học thuật (Copyright & Academic Integrity)

Dự án **XLA Cell Counter** được phát triển độc lập bởi **Nguyễn Phan Bảo Nguyên** cho mục đích học tập, nghiên cứu Xử lý ảnh số và xây dựng hồ sơ năng lực cá nhân (Personal Portfolio).

* **Nghiêm cấm đạo nhái (No Plagiarism):** Nghiêm cấm mọi hành vi sao chép mã nguồn, chỉnh sửa tên tác giả hoặc sử dụng toàn bộ/một phần dự án này để nộp làm bài tập lớn, đồ án môn học dưới danh nghĩa cá nhân khác.
* **Quyền sử dụng tham khảo:** Bạn được phép xem mã nguồn để tham khảo kiến thức thuật toán. Nếu trích dẫn hoặc phát triển tiếp từ mã nguồn này, bắt buộc phải giữ nguyên thông tin tác giả gốc (Attribution) và dẫn nguồn trực tiếp về repository này.

---

## 👨‍💻 Tác giả (Author)

* **Họ và tên:** Nguyễn Phan Bảo Nguyên
* **GitHub:** [@ShadowPacket07](https://github.com/ShadowPacket07)
* **Email:** iuhcoderk20@gmail.com

---
*© 2026 Nguyễn Phan Bảo Nguyên. All Rights Reserved.*