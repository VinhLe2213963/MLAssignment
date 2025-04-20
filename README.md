# MLAssignment

# Data

# Cách hiện thực

Dự án này trình bày cách triển khai thuật toán PCA để giảm chiều dữ liệu và sau đó sử dụng Logistic Regression và một phương pháp tiếp cận khác là SVM cho bài toán phân loại MNIST. Toàn bộ quá trình được thực hiện trong file Jupyter Notebook (`pca_and_svm.ipynb`).

## Giới thiệu

PCA là một kỹ thuật giảm chiều dữ liệu phổ biến, giúp chuyển đổi tập dữ liệu ban đầu thành một tập dữ liệu mới có số chiều nhỏ hơn trong khi vẫn giữ lại phần lớn thông tin quan trọng (phương sai). Logistics Regression là một mô hình tuyến tính cơ bản nhưng hiệu quả cho bài toán phân loại đa lớp. SVM là một thuật toán học máy giám sát mạnh mẽ, thường được sử dụng cho các bài toán phân loại và hồi quy.

Notebook này hướng dẫn từng bước thực hiện:
1.  Tiền xử lý dữ liệu.
2.  Áp dụng PCA để giảm số chiều và huấn luyện mô hình Logistics Regression trên dữ liệu đã xử lý.
3.  Huấn luyện mô hình bằng phương pháp SVM
4.  Đánh giá hiệu suất của 2 mô hình.

## Công nghệ sử dụng

*   Python 3
*   Jupyter Notebook
*   Google Colab
*   Scikit-learn 
*   Numpy
*   Matplotlib / Seaborn

## Dữ liệu

Notebook này có thể sử dụng tập dữ liệu MNIST được tải trực tiếp trong code. Vui lòng kiểm tra các cell đầu tiên của notebook (`pca_and_svm.ipynb`) để biết chi tiết về nguồn dữ liệu và cách tải/cung cấp dữ liệu.

## Cách sử dụng

Để chạy project này, bạn cần sử dụng Google Colab. Thực hiện theo các bước sau:

1.  **Tải về Notebook:** Tải file `implementation.ipynb` về máy tính của bạn.
2.  **Mở Google Colab:** Truy cập [https://colab.research.google.com/](https://colab.research.google.com/) và đăng nhập bằng tài khoản Google của bạn.
3.  **Tải Notebook lên Colab:**
    *   Chọn `Tệp` (File) > `Tải sổ tay lên...` (Upload notebook...).
    *   Chọn file `implementation.ipynb` bạn vừa tải về.
4.  **Chuẩn bị Dữ liệu (Nếu cần):**
    *   Nếu notebook yêu cầu tải lên file dữ liệu, hãy đảm bảo bạn đã tải file đó lên Google Drive và mount Drive vào Colab, hoặc sử dụng tính năng tải file trực tiếp của Colab. Các hướng dẫn cụ thể thường có trong các cell đầu tiên của notebook.
5.  **Chạy các Cell:**
    *   Thực thi từng cell code trong notebook theo thứ tự từ trên xuống dưới.
    *   Bạn có thể chạy từng cell bằng cách nhấn nút `Play` (▶️) bên cạnh cell hoặc sử dụng phím tắt `Shift + Enter`.
    *   **Lưu ý:** Notebook có thể chứa các cách triển khai hoặc bước thực hiện khác nhau. Hãy đọc kỹ mô tả trong từng cell hoặc các phần văn bản (markdown) để hiểu rõ mục đích và kết quả mong đợi của từng bước.

## Kết quả

1. **PCA kết hợp với Logistics Regression:** 92.04%
2. **Support Vector Machine với RBF Kernel:** 98.23%

## Đóng góp

Nếu bạn muốn đóng góp hoặc cải thiện project này, vui lòng tạo Pull Request hoặc mở Issue.
