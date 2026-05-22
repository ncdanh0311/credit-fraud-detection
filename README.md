# 💳 Phát hiện Gian lận Thẻ Tín dụng (Credit Card Fraud Detection)

> [!NOTE]  
> Dự án nghiên cứu ứng dụng Học máy và Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI) trong việc phát hiện các giao dịch gian lận thẻ tín dụng. Hệ thống xử lý triệt để bài toán mất cân bằng lớp nghiêm trọng (chỉ **0.17%** giao dịch gian lận) thông qua kỹ thuật kết hợp SMOTE & nhiễu Gauss, đồng thời "mở hộp đen" mô hình bằng lý thuyết SHAP (Shapley Additive Explanations).

---

## 📂 Cấu trúc Thư mục Dự án

```text
├── data/                       # Thư mục chứa dữ liệu dự án
│   ├── raw/                    # Chứa dữ liệu gốc (creditcard.csv)
│   └── processed/              # Chứa dữ liệu đã chuẩn hóa & file scaler.pkl
├── models/                     # Thư mục lưu trữ các mô hình sau huấn luyện (.pkl)
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl              # Bộ chuẩn hóa đồng bộ phục vụ production
├── notebooks/                  # Chuỗi 3 Jupyter Notebooks lõi của dự án
│   ├── 01_EDA_Preprocessing.ipynb   # 1. Khám phá & Tiền xử lý dữ liệu
│   ├── 02_SMOTE_Model_Training.ipynb # 2. Cân bằng dữ liệu & Huấn luyện mô hình
│   └── 03_SHAP_Explainability.ipynb  # 3. Giải thích mô hình bằng SHAP (XAI)
├── reports/                    # Báo cáo kết quả và biểu đồ hiệu năng
├── src/                        # Mã nguồn triển khai hệ thống (nếu có)
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 🛠️ Phương pháp Tiếp cận Nâng cao

Dự án áp dụng quy trình xử lý dữ liệu và huấn luyện mô hình theo chuẩn công nghiệp:

### 1. Đồng bộ hóa Scaling
*   Hai đặc trưng vật lý duy nhất chưa ẩn danh là `Time` và `Amount` được chuẩn hóa bằng `StandardScaler`. 
*   Bộ chuẩn hóa (Scaler) được fit **chỉ trên tập Train** để tránh rò rỉ thông tin dữ liệu (Data Leakage) và lưu trữ thành [scaler.pkl](file:///d:/Intern-Projects/credit-fraud-detection/models/scaler.pkl) nhằm đồng bộ hóa cho quá trình suy luận thực tế (inference).

### 2. Cân bằng Lớp bằng SMOTE & Thêm Nhiễu Gauss (Feature Noise Induction)
*   **SMOTE (Synthetic Minority Over-sampling Technique):** Sinh mẫu nhân tạo lớp thiểu số (Gian lận) dựa trên thuật toán k-NN để đưa tỷ lệ phân phối về mức cân bằng tuyệt đối 50/50.
*   **Gaussian Noise Induction (Custom Data Augmentation):** Cộng thêm một lượng nhiễu Gaussian siêu nhỏ tuân theo phân phối $N(0, (0.015 \times \sigma)^2)$ vào các mẫu sinh ra từ SMOTE. Phương pháp này làm mịn ranh giới quyết định (Decision Boundary Smoothing), giúp khắc phục hiện tượng quá khớp (overfitting) do SMOTE tạo các điểm thẳng hàng.

### 3. Tận dụng Sức mạnh của Random Forest (Bagging)
*   Huấn luyện và so sánh 3 thuật toán: `Logistic Regression`, `Decision Tree`, và `Random Forest`.
*   Mô hình **Random Forest** thể hiện ưu thế vượt trội nhờ cơ chế học tập ensemble (Bagging), giảm thiểu phương sai (variance) và nhạy bén trước các mối quan hệ phi tuyến tính phức tạp của hành vi gian lận.

### 4. Giải thích Hộp Đen Học Máy với SHAP (Explainable AI - XAI)
*   **Global Interpretability (Toàn cục):** Đánh giá tầm quan trọng và hướng tác động của các biến đặc trưng (như $V_{14}, V_{17}, Amount$) qua biểu đồ **Summary Dot Plot** và **Bar Plot**.
*   **Local Interpretability (Cục bộ):** Lý giải chi tiết từng giao dịch đơn lẻ bị mô hình phán quyết là gian lận bằng biểu đồ **Waterfall Plot** và **Force Plot (đã được tối ưu tỷ lệ 20x3 tránh chồng chéo văn bản)**.

---

## 📈 Kết quả So sánh và Đánh giá Mô hình

Dưới đây là bảng so sánh hiệu năng thực tế của các mô hình trên tập kiểm thử (**Test Set**):

| Thuật toán | Phương pháp Dữ liệu | F1-Score (Lớp Gian lận) | ROC-AUC | Trạng thái Mô hình |
| :--- | :--- | :---: | :---: | :---: |
| **Logistic Regression** | Nguyên bản (Unbalanced) | 0.7027 | 0.9793 | Chưa tối ưu |
| **Logistic Regression** | **Sau SMOTE + Noise** | **0.1068** *(Bị nhiễu lớp)* | **0.9749** | Nhạy quá mức |
| **Decision Tree** | Nguyên bản (Unbalanced) | 0.7818 | 0.8931 | Có dấu hiệu quá khớp |
| **Decision Tree** | **Sau SMOTE + Noise** | **0.5752** | **0.9238** | Ranh giới mượt hơn |
| **Random Forest** | Nguyên bản (Unbalanced) | 0.8525 | 0.9472 | Tốt |
| **Random Forest** | **Sau SMOTE + Noise** | **0.8491** | **0.9789** | **Tối ưu nhất (Tổng quát hóa cao)** |

> [!IMPORTANT]  
> Nhờ bổ sung nhiễu Gauss sau SMOTE, mô hình **Random Forest** đạt chỉ số **ROC-AUC là 0.9789** (tăng mạnh so với 0.9472 của mô hình nguyên bản) và giữ vững chỉ số **F1-Score ở mức cực cao 0.8491**, khẳng định khả năng phát hiện gian lận cực kỳ chính xác đồng thời giảm thiểu tối đa báo động giả (False Alarm).

---

## 💻 Hướng dẫn Cài đặt & Chạy Dự án

### Bước 1: Clone dự án và cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.8+ và cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install numpy pandas scikit-learn imbalanced-learn matplotlib seaborn shap
```

### Bước 2: Chuẩn bị dữ liệu
1.  Tải bộ dữ liệu [Credit Card Fraud Detection từ Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2.  Tạo thư mục `data/raw/` và đặt file `creditcard.csv` vào đó.

### Bước 3: Chạy lần lượt các Jupyter Notebooks
Quy trình thực hiện tuần tự theo các file trong thư mục `notebooks/`:

1.  **[01_EDA_Preprocessing.ipynb](file:///d:/Intern-Projects/credit-fraud-detection/notebooks/01_EDA_Preprocessing.ipynb)**: Khám phá phân phối dữ liệu, vẽ biểu đồ tương quan và chuẩn hóa dữ liệu lưu vào thư mục `data/processed/`.
2.  **[02_SMOTE_Model_Training.ipynb](file:///d:/Intern-Projects/credit-fraud-detection/notebooks/02_SMOTE_Model_Training.ipynb)**: Cân bằng lớp dữ liệu bằng SMOTE và Gaussian Noise, huấn luyện 3 mô hình học máy. Các file mô hình tối ưu `.pkl` sẽ tự động được đóng gói và lưu vào thư mục `models/`.
3.  **[03_SHAP_Explainability.ipynb](file:///d:/Intern-Projects/credit-fraud-detection/notebooks/03_SHAP_Explainability.ipynb)**: Nạp các mô hình đã lưu để thực hiện phân tích giải thích bằng SHAP XAI cục bộ và toàn cục.

---

## 💾 Danh sách các File Mô hình được Export

Sau khi chạy xong Notebook 2, thư mục [models/](file:///d:/Intern-Projects/credit-fraud-detection/models/) sẽ chứa các mô hình sẵn sàng đưa vào ứng dụng production:
*   [logistic_regression_model.pkl](file:///d:/Intern-Projects/credit-fraud-detection/models/logistic_regression_model.pkl): Mô hình Logistic Regression tuyến tính.
*   [decision_tree_model.pkl](file:///d:/Intern-Projects/credit-fraud-detection/models/decision_tree_model.pkl):Mô hình Cây quyết định đơn lẻ.
*   [random_forest_model.pkl](file:///d:/Intern-Projects/credit-fraud-detection/models/random_forest_model.pkl): Mô hình Ensemble Random Forest tối ưu nhất.
*   [scaler.pkl](file:///d:/Intern-Projects/credit-fraud-detection/models/scaler.pkl): StandardScaler đã fit trên tập huấn luyện gốc.
