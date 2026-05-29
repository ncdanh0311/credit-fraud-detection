# 💳 Phát hiện Gian lận Thẻ Tín dụng (Credit Card Fraud Detection)
---
## 📂 Cấu trúc Thư mục Dự án
```text
├── data/                       # Thư mục chứa dữ liệu dự án
│   ├── raw/                    # Chứa dữ liệu gốc (creditcard.csv)
│   └── processed/              # Chứa dữ liệu train/test đã được chuẩn hóa
├── models/                     # Thư mục lưu trữ các bộ chuẩn hóa và mô hình (.joblib, .keras)
│   ├── scaler_amount.joblib    # Bộ chuẩn hóa riêng biệt cho Amount
│   ├── scaler_time.joblib      # Bộ chuẩn hóa riêng biệt cho Time
│   ├── LogisticRegression_NoSMOTE.joblib / LogisticRegression_SMOTE.joblib
│   ├── DecisionTree_NoSMOTE.joblib / DecisionTree_SMOTE.joblib
│   ├── RandomForest_NoSMOTE.joblib / RandomForest_SMOTE.joblib
│   ├── 1D-CNN_NoSMOTE.keras / 1D-CNN_SMOTE.keras (kèm file .weights.h5)
├── notebooks/                  # Chuỗi 3 Jupyter Notebooks lõi của dự án
│   ├── 01_EDA_Preprocessing.ipynb   # 1. Khám phá dữ liệu & Tiền xử lý (Separate Scaling, SMOTE)
│   ├── 02_SMOTE_Model_Training.ipynb # 2. Huấn luyện & Xuất 8 cấu hình mô hình (ML & DL CNN)
│   └── 03_SHAP_Explainability.ipynb  # 3. Giải thích mô hình tối ưu bằng SHAP XAI (Waterfall, Force Plot)
├── outputs/                    # Thư mục lưu trữ kết quả phân tích
│   ├── plots/                  # Thư mục lưu các biểu đồ ROC & F1-Score so sánh
│   └── shap/                   # Biểu đồ SHAP toàn cục & Force Plot cục bộ dạng HTML
├── src/                        # Mã nguồn Streamlit App phục vụ demo/inference
│   └── app.py                  # Web Dashboard phát hiện gian lận tích hợp XAI & SQL Server Logging
└── README.md                   # Tài liệu hướng dẫn dự án (File này)
```
---
## 🛠️ Phương pháp Tiếp cận Lõi (Master Pipeline)
Dự án áp dụng quy trình xử lý dữ liệu chuẩn hóa công nghiệp, đảm bảo tính đồng bộ tuyệt đối giữa mô hình huấn luyện và hệ thống chạy thực tế (Production):
### 1. Đồng bộ hóa Scaling riêng biệt (Separate Scaling)
*   Để chống rò rỉ thông tin dữ liệu (Data Leakage) và phục vụ tốt nhất cho giao diện Web Dashboard, hai đặc trưng vật lý duy nhất chưa ẩn danh là `Time` và `Amount` được chuẩn hóa độc lập bằng hai bộ chuẩn hóa riêng biệt: **`scaler_amount`** và **`scaler_time`** (`StandardScaler`).
*   Các đặc trưng sau khi chuẩn hóa được sắp xếp lên đầu DataFrame theo thứ tự: `scaled_amount`, `scaled_time` ở vị trí index 0 và 1, theo sau bởi 28 biến PCA ẩn danh từ `V1` đến `V28` (Tạo cấu trúc 30 cột đồng nhất).
### 2. Giải quyết Mất cân bằng Lớp nghiêm trọng bằng SMOTE
*   **SMOTE (Synthetic Minority Over-sampling Technique):** Tự động sinh mẫu nhân tạo lớp thiểu số (Gian lận - Class 1) dựa trên các láng giềng k-NN trên tập Train để đưa phân phối nhãn về mức cân bằng lý tưởng **50:50**, giúp các mô hình không bị thiên vị nhóm đa số.
### 3. Huấn luyện & So sánh 8 cấu hình mô hình (ML & DL)
*   Hệ thống xây dựng và so sánh đồng thời 4 thuật toán: `Logistic Regression`, `Decision Tree`, `Random Forest`, và mạng nơ-ron tích chập **`1D-CNN`** (Deep Learning).
*   Mỗi thuật toán được huấn luyện trên 2 tập dữ liệu: **Dữ liệu gốc (NoSMOTE)** và **Dữ liệu cân bằng (SMOTE)** để đánh giá định lượng và trực quan hóa so sánh hiệu năng.
### 4. Giải thích Hộp Đen Học Máy với SHAP (Explainable AI - XAI)
*   **Global Interpretability (Toàn cục):** Đánh giá các biến số đóng vai trò quyết định cấu trúc mô hình qua biểu đồ **Summary Dot Plot** và **Bar Plot** (với $V_{14}$, $V_{17}$ là các biến tác động mạnh nhất).
*   **Local Interpretability (Cục bộ):** Trực quan hóa chi tiết từng giao dịch đơn lẻ bị mô hình phán quyết là gian lận qua biểu đồ **Waterfall Plot** và **Force Plot** tương tác, chỉ rõ lực đẩy tăng/giảm xác suất rủi ro.
---
## 📈 Kết quả Đánh giá Mô hình trên tập kiểm thử (Test Set)
Khi áp dụng SMOTE, hiệu năng phát hiện gian lận (đặc biệt là chỉ số F1-Score của lớp Gian lận và ROC-AUC) được cải thiện vượt trội:
*   **Random Forest (SMOTE):** Đạt khả năng tổng quát hóa cao nhất, giữ chỉ số F1-Score cực tốt đồng thời giảm thiểu báo động giả.
*   **1D-CNN (SMOTE):** Có tốc độ học và trích xuất đặc trưng không gian siêu việt nhờ các lớp Convolutional, đạt F1-Score và AUC tối ưu khi dữ liệu được cân bằng hoàn hảo.
---
## 💻 Hướng dẫn Cài đặt & Chạy Dự án
### Bước 1: Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.8+ và các thư viện cần thiết bằng lệnh:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn shap joblib tensorflow streamlit streamlit-shap pyodbc
```
### Bước 2: Chuẩn bị dữ liệu
1.  Tải bộ dữ liệu [Credit Card Fraud Detection từ Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  Tạo thư mục `data/raw/` và đặt file `creditcard.csv` vào đó.
### Bước 3: Chạy lần lượt các Jupyter Notebooks
Tất cả các notebook đã được tích hợp cơ chế tự động nhận diện và kết nối Google Drive (Colab) hoặc Local IDE linh hoạt:
1.  **[01_EDA_Preprocessing.ipynb](file:///d:/Intern-Projects/credit-fraud-detection%20-%20Copy/notebooks/01_EDA_Preprocessing.ipynb)**: Thực hiện khám phá dữ liệu, tương quan, chuẩn hóa riêng biệt 2 scaler và chạy SMOTE xuất ra `data/processed/`.
2.  **[02_SMOTE_Model_Training.ipynb](file:///d:/Intern-Projects/credit-fraud-detection%20-%20Copy/notebooks/02_SMOTE_Model_Training.ipynb)**: Huấn luyện và đóng gói 8 cấu hình mô hình lưu trữ trực tiếp vào `models/`, vẽ đồ thị ROC & F1-Score.
3.  **[03_SHAP_Explainability.ipynb](file:///d:/Intern-Projects/credit-fraud-detection%20-%20Copy/notebooks/03_SHAP_Explainability.ipynb)**: Nạp mô hình tối ưu `RandomForest_SMOTE.joblib` để giải thích nghiệp vụ XAI cục bộ và toàn cục kèm slider What-If tương tác.
### Bước 4: Khởi động Web Dashboard Streamlit
Chạy lệnh sau tại thư mục gốc của project để mở giao diện dashboard phát hiện gian lận thời gian thực:
```bash
streamlit run src/app.py
```
*   **Tính năng chính:** Chọn 1 trong 4 mô hình cốt lõi, tự động điền mẫu giao dịch nghi ngờ/hợp lệ, phân tích What-If tùy chỉnh các biến, trực quan hóa SHAP Force Plot thời gian thực, và lưu trữ nhật ký giao dịch tự động xuống SQL Server (kèm offline fallback).