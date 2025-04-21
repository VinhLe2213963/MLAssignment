# MLAssignment

## Phân chia công việc

| STT | Họ và tên           | MSSV    | Vai trò                                                                                                                                  | Đóng góp |
| --- | ------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| 1   | Lê Trường Thống   | 2213338 | Phân tích dữ liệu + CNN                                                                                            | 100%               |
| 2   | Kiều Tâm Hậu        | 2210961 | Support Vector Machine                                                                                        | 100%                 |
| 3   | Lê Hoàng Khánh Vinh | 2213963 | PCA + Logistics Regression                                                                                        | 100%                 |
| 4   | Lê Quốc Huy | 2211194 | Phân tích dữ liệu                                                                                        | 100%                 |

# Giới thiệu đề tài

Trong lĩnh vực thị giác máy tính và nhận dạng hình ảnh, việc xử lý và phân loại dữ liệu hình ảnh có độ phân giải cao là một thách thức đáng kể, đòi hỏi các phương pháp hiệu quả trong việc giảm chiều dữ liệu và xây dựng mô hình phân loại chính xác. Tập dữ liệu MNIST – bao gồm các hình ảnh chữ số viết tay từ 0 đến 9 – là một trong những bộ dữ liệu tiêu chuẩn được sử dụng rộng rãi để đánh giá hiệu quả của các thuật toán học máy trong bài toán phân loại hình ảnh.

Đề tài này tập trung vào việc triển khai và so sánh hai giải pháp học máy nhằm giải quyết bài toán phân loại chữ số viết tay trên tập MNIST. Giải pháp đầu tiên sử dụng Principal Components Analysis (PCA) để giảm chiều dữ liệu đầu vào, sau đó áp dụng Logistic Regression để xây dựng mô hình phân loại. Giải pháp này hướng đến việc cải thiện hiệu suất huấn luyện và giảm thiểu độ phức tạp tính toán bằng cách loại bỏ các đặc trưng dư thừa, đồng thời vẫn giữ lại những thông tin quan trọng nhất trong dữ liệu.

Giải pháp thứ hai áp dụng trực tiếp Support Vector Machine (SVM) – một mô hình học máy mạnh mẽ với khả năng tìm kiếm siêu mặt phân tách tối ưu giữa các lớp dữ liệu. SVM được đánh giá cao trong các bài toán phân loại nhị phân và đa lớp, đặc biệt khi dữ liệu không tuyến tính có thể được ánh xạ vào không gian đặc trưng cao hơn thông qua các hàm kernel.

Mục tiêu của đề tài là đánh giá hiệu quả của từng giải pháp về độ chính xác phân loại, thời gian huấn luyện và khả năng tổng quát hóa mô hình. Từ đó, rút ra nhận xét về ưu nhược điểm của từng phương pháp trong bối cảnh xử lý dữ liệu hình ảnh độ phân giải cao như MNIST.

# Data

Trong bài tập lớn này, nhóm lựa chọn sử dụng tập dữ liệu **MNIST**, một trong những bộ dữ liệu phổ biến nhất trong lĩnh vực thị giác máy tính, đặc biệt là các bài toán phân loại ảnh. **MNIST** bao gồm các hình ảnh chữ số viết tay từ 0 đến 9, được thu thập và chuẩn hóa để phục vụ cho các thuật toán học máy.

Cụ thể, tập dữ liệu **MNIST** được chia thành hai phần:
- Tập huấn luyện (training set) với 60.000 mẫu,
- Tập kiểm tra (test set) với 10.000 mẫu.

Mỗi mẫu là một ảnh grayscale có kích thước 28x28 pixel, với mỗi pixel mang giá trị nguyên trong khoảng từ 0 đến 255. Giá trị pixel càng thấp thì màu sắc càng tối (0 là đen hoàn toàn), ngược lại giá trị càng cao thì màu càng sáng (255 là trắng hoàn toàn). Nhờ được gán nhãn đầy đủ, mỗi ảnh đều đi kèm với thông tin về chữ số tương ứng, tạo điều kiện thuận lợi cho việc huấn luyện và đánh giá mô hình theo phương pháp supervised learning.

Tuy nhiên, do đặc điểm của chữ viết tay, dữ liệu trong MNIST chứa nhiều mẫu chữ số có hình dạng không rõ ràng, dễ gây nhầm lẫn — ví dụ giữa các chữ số như "1" và "7" hoặc "5" và "8". Chính vì vậy, trước khi lựa chọn mô hình học máy, nhóm đã thực hiện các phân tích dữ liệu sơ bộ nhằm hiểu rõ hơn về đặc điểm của tập dữ liệu, xác định những mẫu dễ gây nhầm lẫn và đánh giá sự phân bố dữ liệu. Các phân tích này được thực hiện trong file `data_analysis.ipynb`.

# Cách hiện thực

Dự án này trình bày cách triển khai thuật toán PCA để giảm chiều dữ liệu và sau đó sử dụng Logistic Regression và một phương pháp tiếp cận khác là SVM cho bài toán phân loại MNIST. Toàn bộ quá trình được thực hiện trong file Jupyter Notebook (`implementation.ipynb`).

Ngoài các thuật toán, mô hình trước đó, phần mở rộng của bài tập lớn này là mạng nơ-ron tích chập (Convolutional Neutral Network - CNN) được sử dụng để nhận diện hình ảnh, được hiện thực trong file Python (`convolution_nn.py`). 

## Giới thiệu

PCA là một kỹ thuật giảm chiều dữ liệu phổ biến, giúp chuyển đổi tập dữ liệu ban đầu thành một tập dữ liệu mới có số chiều nhỏ hơn trong khi vẫn giữ lại phần lớn thông tin quan trọng (phương sai). Logistics Regression là một mô hình tuyến tính cơ bản nhưng hiệu quả cho bài toán phân loại đa lớp. SVM là một thuật toán học máy giám sát mạnh mẽ, thường được sử dụng cho các bài toán phân loại và hồi quy. CNN là mô hình phổ biến được sử dụng trong bài toán phân loại hình ảnh (Image Classification).

Notebook này hướng dẫn từng bước thực hiện:
1.  Tiền xử lý dữ liệu.
2.  Áp dụng PCA để giảm số chiều và huấn luyện mô hình Logistics Regression trên dữ liệu đã xử lý.
3.  Huấn luyện mô hình bằng phương pháp SVM
4.  Xây dựng mô hình CNN để dự đoán các hình ảnh
5.  Đánh giá hiệu suất của các mô hình.

## Công nghệ sử dụng

*   Python 3
*   Jupyter Notebook
*   Google Colab
*   Scikit-learn 
*   Numpy
*   Matplotlib / Seaborn
*   Pandas
*   Thundersvm

## Dữ liệu

Notebook này có thể sử dụng tập dữ liệu MNIST được tải trực tiếp trong code. Vui lòng kiểm tra các cell đầu tiên của notebook (`implementation.ipynb`) để biết chi tiết về nguồn dữ liệu và cách tải/cung cấp dữ liệu.

Có thể sử dụng dataset thủ công dạng (`.csv`) được cung cấp sẵn trong file nén (`mnist_data.zip`). Cần sử dụng phương thức `read_csv` của thư viện Pandas để đọc được các file dữ liệu này.

## Cách sử dụng

Để chạy project này với các mô hình PCA kết hợp Logistic Regression và SVM, bạn cần sử dụng Google Colab. Thực hiện theo các bước sau:

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

Đối với mô hình CNN, có thể chạy trực tiếp trên máy local bằng lệnh `python3 convolution_nn.py`, có thể cần thiết phải sử dụng lệnh `pip install` để cài đặt một số các thư viện cần thiết.  

## Kết quả

1. **PCA kết hợp với Logistics Regression:** 92.04%
2. **Support Vector Machine với RBF Kernel:** 98.23%
3. **Convolutional Neutral Network:** 97.08%

## Đóng góp

Nếu bạn muốn đóng góp hoặc cải thiện project này, vui lòng tạo Pull Request hoặc mở Issue.
