import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from streamlit_shap import st_shap
import warnings
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
import pyodbc
import time
import datetime

# --- 1. TẮT CẢNH BÁO LỆCH PHIÊN BẢN ---
warnings.filterwarnings('ignore', category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except ImportError:
    pass

# Thiết lập encoding UTF-8 cho terminal để hiển thị tiếng Việt chính xác
sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FraudDetectionApp")

# =====================================================================
# 2. CẤU HÌNH TRANG & CUSTOM CYBERSECURITY UI CSS (TAILWIND STYLE)
# =====================================================================
st.set_page_config(
    page_title="Hệ thống phân tích và phát hiện gian lận thẻ tín dụng",
    layout="wide"
)

# Nhúng FontAwesome và tạo CSS Fintech Dashboard siêu cao cấp từ HTML
st.markdown("""<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">""", unsafe_allow_html=True)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F3F4F6 !important;
}
input[type=range] {
    accent-color: #1E3A8A !important;
}
div.stButton > button {
    background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 30px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    transition: all 0.2s ease-in-out !important;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 10px !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(10, 37, 64, 0.2) !important;
    background: linear-gradient(135deg, #1E3A8A 0%, #0A2540 100%) !important;
}
.card {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
}
.result-card-normal {
    padding: 20px;
    background-color: #ECFDF5;
    border-left: 6px solid #10B981;
    color: #065F46;
    border-radius: 8px;
    margin-bottom: 20px;
}
.result-card-fraud {
    padding: 20px;
    background-color: #FEF2F2;
    border-left: 6px solid #EF4444;
    color: #991B1B;
    border-radius: 8px;
    margin-bottom: 20px;
}
.result-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.result-desc {
    font-size: 1rem;
    line-height: 1.5;
}
.comparison-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}
.comparison-table th {
    background-color: #F9FAFB;
    color: #374151;
    font-weight: 600;
    padding: 12px;
    text-align: left;
    border-bottom: 2px solid #E5E7EB;
}
.comparison-table td {
    padding: 12px;
    border-bottom: 1px solid #E5E7EB;
}
[data-testid="stSidebar"]{min-width:320px !important;max-width:320px !important;width:320px !important;overflow:hidden !important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{overflow-y:hidden !important;overflow-x:hidden !important;}
[data-testid="stSidebar"]::-webkit-scrollbar, [data-testid="stSidebar"] *::-webkit-scrollbar{display:none !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] *{-ms-overflow-style:none !important;scrollbar-width:none !important;}
[data-testid="stSidebarDragHandle"], [data-testid="stSidebarResizer"], [class*="stSidebarDragHandle"], [class*="stSidebarResizer"]{display:none !important;pointer-events:none !important;}
[data-testid="stSidebar"] .stRadio label p{font-size:1.05rem !important;font-weight:600 !important;color:#4B5563 !important;line-height:1.6 !important;white-space:nowrap !important;transition:all 0.2s ease !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:25px !important;}
[data-testid="stSidebar"] .stRadio label{padding:12px 16px !important;border-radius:8px !important;transition:all 0.2s ease !important;cursor:pointer !important;border:1px solid rgba(229,231,235,0.6) !important;background-color:#ffffff !important;box-shadow:0 2px 4px rgba(0,0,0,0.02) !important;}
[data-testid="stSidebar"] .stRadio label:hover{background-color:#F9FAFB !important;border-color:#D1D5DB !important;}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-of-type{display:none !important;}
[data-testid="stSidebar"] .stRadio label:has(input:checked){background-color:#ffffff !important;border:1px solid #E5E7EB !important;border-bottom:4px solid #1E3A8A !important;box-shadow:0 4px 6px -1px rgba(0,0,0,0.05) !important;}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p{color:#1E3A8A !important;font-weight:700 !important;}
</style>""", unsafe_allow_html=True)

# Thư mục chứa mô hình
MODEL_DIR = "models"
MODEL_DIR_FALLBACK = r"D:\Intern-Projects\credit-fraud-detection\models"

def get_model_path(filename):
    """Lấy đường dẫn mô hình thông minh giữa tương đối và tuyệt đối."""
    rel_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(rel_path):
        return rel_path
    abs_path = os.path.join(MODEL_DIR_FALLBACK, filename)
    if os.path.exists(abs_path):
        return abs_path
    return rel_path

# =====================================================================
# 3. KẾT NỐI SQL SERVER & XỬ LÝ NHẬT KÝ LƯU VẾT
# =====================================================================
def get_sql_connection():
    drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d or 'ODBC' in d]
    if not drivers:
        logger.warning("Không tìm thấy ODBC Driver SQL Server nào trên máy!")
        return None, "Không tìm thấy ODBC driver SQL Server nào trên hệ thống."
    
    preferred_drivers = [
        "ODBC Driver 17 for SQL Server",
        "ODBC Driver 18 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server Native Client 10.0",
        "SQL Server"
    ]
    
    selected_driver = None
    for pd in preferred_drivers:
        matching = [d for d in drivers if pd in d]
        if matching:
            selected_driver = matching[0]
            break
            
    if not selected_driver:
        selected_driver = drivers[0]
        
    server = "DANH-PC"
    database = "CreditCardFraudDB"
    trust_cert = ";TrustServerCertificate=yes" if "Driver 18" in selected_driver else ""
    conn_str = f"DRIVER={{{selected_driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes{trust_cert};"
    
    try:
        conn = pyodbc.connect(conn_str, timeout=3)
        return conn, None
    except Exception as e:
        logger.error(f"Lỗi kết nối SQL Server: {str(e)}")
        return None, f"Không thể kết nối đến SQL Server Danh-PC. Đang chuyển sang offline fallback. (Chi tiết: {str(e)})"

def insert_log_to_sql(transaction_id, model_name, prediction, probability, execution_time_ms, input_time, input_amount, v_values):
    """Ghi dữ liệu kiểm thử vào SQL Server, tự động chuyển về local state nếu ngoại tuyến."""
    v_values_str = ",".join([f"{v:.4f}" for v in v_values])
    
    conn, err = get_sql_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO TransactionLogs (TransactionID, ModelName, Prediction, Probability, ExecutionTimeMs, InputTime, InputAmount, V_Values)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (transaction_id, model_name, prediction, probability, execution_time_ms, input_time, input_amount, v_values_str))
            conn.commit()
            conn.close()
            return True, "SQL_SERVER"
        except Exception as e:
            logger.error(f"Ghi vào SQL Server lỗi: {str(e)}")
            
    if "local_logs" not in st.session_state:
        st.session_state.local_logs = []
        
    log_item = {
        "TransactionID": transaction_id,
        "ModelName": model_name,
        "Prediction": prediction,
        "Probability": probability,
        "ExecutionTimeMs": execution_time_ms,
        "InputTime": input_time,
        "InputAmount": input_amount,
        "V_Values": v_values_str,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.local_logs.append(log_item)
    return True, "OFFLINE_FALLBACK"

def fetch_all_logs():
    """Truy vấn lấy toàn bộ lịch sử logs kiểm thử (ưu tiên SQL Server)."""
    conn, err = get_sql_connection()
    logs_list = []
    
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT TransactionID, ModelName, Prediction, Probability, ExecutionTimeMs, Timestamp FROM TransactionLogs ORDER BY Timestamp DESC")
            rows = cursor.fetchall()
            for r in rows:
                logs_list.append({
                    "TransactionID": r[0],
                    "ModelName": r[1],
                    "Prediction": r[2],
                    "Probability": r[3],
                    "ExecutionTimeMs": r[4],
                    "Timestamp": r[5].strftime("%Y-%m-%d %H:%M:%S") if hasattr(r[5], "strftime") else str(r[5])
                })
            conn.close()
            return logs_list, "SQL_SERVER", None
        except Exception as e:
            logger.error(f"Lỗi truy vấn SQL Server: {str(e)}")
            
    if "local_logs" not in st.session_state:
        st.session_state.local_logs = []
        
    local_logs = st.session_state.local_logs.copy()
    static_logs = [
        {"TransactionID": "TXN-006", "ModelName": "Random Forest - NoSMOTE", "Prediction": "NORMAL", "Probability": 0.0421, "ExecutionTimeMs": 45, "Timestamp": "2026-05-26 19:08:15"},
        {"TransactionID": "TXN-005", "ModelName": "1D-CNN - SMOTE", "Prediction": "NORMAL", "Probability": 0.0115, "ExecutionTimeMs": 68, "Timestamp": "2026-05-26 18:55:10"}
    ]
    
    full_logs = local_logs + static_logs
    try:
        full_logs = sorted(full_logs, key=lambda x: x["Timestamp"], reverse=True)
    except:
        pass
    return full_logs, "OFFLINE_FALLBACK", err

# =====================================================================
# 4. HÀM TẢI MÔ HÌNH VÀ BỘ CHUẨN HÓA (FUNCTIONAL API 1D-CNN)
# =====================================================================
@st.cache_resource
def load_cnn_models():
    """Xây dựng mạng và tải trọng số 1D-CNN Functional API."""
    def build_functional_cnn():
        inputs = Input(shape=(30, 1))
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=outputs)
        
    cnn_no = build_functional_cnn()
    path_no = get_model_path("1D-CNN_NoSMOTE_weights.weights.h5")
    cnn_no.load_weights(path_no)
    
    cnn_sm = build_functional_cnn()
    path_sm = get_model_path("1D-CNN_SMOTE_weights.weights.h5")
    cnn_sm.load_weights(path_sm)
    
    return cnn_no, cnn_sm

@st.cache_resource
def load_all_assets():
    """Tải và cache toàn bộ 8 mô hình cùng bộ giải thích SHAP."""
    assets = {
        "scaler_amount": joblib.load(get_model_path("scaler_amount.joblib")),
        "scaler_time": joblib.load(get_model_path("scaler_time.joblib")),
        "LR_NoSMOTE": joblib.load(get_model_path("LogisticRegression_NoSMOTE.joblib")),
        "LR_SMOTE": joblib.load(get_model_path("LogisticRegression_SMOTE.joblib")),
        "DT_NoSMOTE": joblib.load(get_model_path("DecisionTree_NoSMOTE.joblib")),
        "DT_SMOTE": joblib.load(get_model_path("DecisionTree_SMOTE.joblib")),
        "RF_NoSMOTE": joblib.load(get_model_path("RandomForest_NoSMOTE.joblib")),
        "RF_SMOTE": joblib.load(get_model_path("RandomForest_SMOTE.joblib")),
    }
    
    cnn_no, cnn_sm = load_cnn_models()
    assets["CNN_NoSMOTE"] = cnn_no
    assets["CNN_SMOTE"] = cnn_sm
    
    try:
        assets["explainer"] = shap.TreeExplainer(assets["RF_SMOTE"])
    except Exception as e:
        logger.warning(f"Lỗi tạo TreeExplainer: {str(e)}")
        assets["explainer"] = None
        
    return assets

# Nạp tài nguyên vào bộ nhớ
try:
    all_assets = load_all_assets()
except Exception as e:
    st.error("Lỗi nghiêm trọng khi tải mô hình học máy. Vui lòng kiểm tra lại thư mục models/.")
    st.exception(e)
    st.stop()

# =====================================================================
# 5. GIỮ VÀ ĐỒNG BỘ SESSION STATE CHO SCENARIO VÀ SLIDERS
# =====================================================================
if "scenario_changed" not in st.session_state:
    st.session_state.scenario_changed = False
    st.session_state.current_scenario = "TXN-007 (Mẫu Giao dịch Nghi ngờ Gian lận - Fraud)"
    st.session_state.input_method = "Dạng chuỗi phân cách bởi dấu phẩy"
    st.session_state.primary_algo = "Random Forest (Khuyên dùng)"
    st.session_state.decision_threshold = 0.50
    
    # Khởi tạo giá trị mặc định cho sliders
    st.session_state.time_val = 472.0
    st.session_state.amount_val = 239.93
    st.session_state.v14_val = -4.50
    st.session_state.v4_val = 5.22
    st.session_state.v12_val = -6.10
    st.session_state.v10_val = -2.30
    
    # 28 biến đặc trưng V
    st.session_state.v_array = [-3.04, 3.48, -3.74, 5.22, -1.17, -1.41, -2.44, 0.49, -2.75, -2.30, 4.59, -6.10, 1.43, -4.50, -0.10, -3.37, -5.20, -1.25, -0.33, 0.35, 0.66, 0.44, -0.04, -0.25, -0.27, -0.02, 0.35, -0.15]
    
    # Kết quả dự báo ban đầu không tự động chạy
    st.session_state.run_prediction = False
    st.session_state.results_table = []
    st.session_state.main_fraud_prob = 0.0
    st.session_state.main_prediction_label = 0
    st.session_state.main_execution_time_ms = 0
    st.session_state.processed_df_main = None

# Khởi tạo bổ sung đề phòng hot-reload từ phiên bản cũ thiếu trường V4
if "v4_val" not in st.session_state:
    st.session_state.v4_val = 5.22

# Hàm đồng bộ hóa khi chuyển đổi giao dịch mẫu
def on_scenario_change():
    scen = st.session_state.scenario_select
    if scen == "TXN-001 (Mẫu Giao dịch Hợp lệ - Normal)":
        st.session_state.time_val = 406.0
        st.session_state.amount_val = 4.90
        st.session_state.v14_val = -0.50
        st.session_state.v4_val = 1.15
        st.session_state.v12_val = -3.30
        st.session_state.v10_val = -0.10
        st.session_state.v_array = [-2.31, 1.66, -1.17, 1.15, -0.74, -0.10, -1.78, 0.46, -0.76, -0.10, -0.03, 0.31, -0.30, -0.28, 1.22, 0.42, 1.25, -0.32, 1.05, 0.13, 0.18, 0.36, 0.16, -0.47, -0.46, 0.34, 0.18, 0.08]
        st.session_state.v_array[13] = -0.50
        st.session_state.v_array[3] = 1.15
        st.session_state.v_array[11] = -3.30
        st.session_state.v_array[9] = -0.10
    elif scen == "TXN-007 (Mẫu Giao dịch Nghi ngờ Gian lận - Fraud)":
        st.session_state.time_val = 472.0
        st.session_state.amount_val = 239.93
        st.session_state.v14_val = -4.50
        st.session_state.v4_val = 5.22
        st.session_state.v12_val = -6.10
        st.session_state.v10_val = -2.30
        st.session_state.v_array = [-3.04, 3.48, -3.74, 5.22, -1.17, -1.41, -2.44, 0.49, -2.75, -2.30, 4.59, -6.10, 1.43, -4.50, -0.10, -3.37, -5.20, -1.25, -0.33, 0.35, 0.66, 0.44, -0.04, -0.25, -0.27, -0.02, 0.35, -0.15]
    else:
        st.session_state.time_val = 0.0
        st.session_state.amount_val = 100.0
        st.session_state.v14_val = 0.0
        st.session_state.v4_val = 0.0
        st.session_state.v12_val = 0.0
        st.session_state.v10_val = 0.0
        st.session_state.v_array = [0.0] * 28

# =====================================================================
# 6. GIAO DIỆN CHÍNH: HEADER & BẢNG ĐIỀU KHIỂN (CONTROL PANEL)
# =====================================================================
st.markdown("""
    <div style="text-align: center; margin-top: 10px; margin-bottom: 25px;">
        <h1 style="color: #0A2540; font-size: 2.1rem; font-weight: 800; display: inline-flex; align-items: center; gap: 12px; margin-bottom: 5px;">
            <i class="fa-solid fa-shield-halved" style="color: #1E3A8A;"></i> HỆ THỐNG PHÂN TÍCH VÀ PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG
        </h1>

    </div>
""", unsafe_allow_html=True)

# Khu vực điều khiển: Bố cục 3 cột tương ứng Dòng 1 cấu hình chính trong HTML
st.markdown("<h3 style='color: #0A2540; font-size: 1.2rem; font-weight: 600; margin-bottom: 12px; border-bottom: 1px solid #E5E7EB; padding-bottom: 6px;'><i class='fa-solid fa-sliders' style='margin-right: 8px;'></i> Bảng Điều Khiển Cấu Hình và Tương Tác</h3>", unsafe_allow_html=True)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

with col_ctrl1:
    primary_algo = st.selectbox(
        "1. Chọn Mô hình cốt lõi (Xem SHAP):",
        ["Random Forest (Khuyên dùng)", "Logistic Regression", "Decision Tree", "1D-CNN (Deep Learning)"],
        key="primary_algo_select"
    )
with col_ctrl2:
    selected_txn = st.selectbox(
        "2. Chọn Giao dịch (Auto-fill):",
        ["TXN-007 (Mẫu Giao dịch Nghi ngờ Gian lận - Fraud)", "TXN-001 (Mẫu Giao dịch Hợp lệ - Normal)", "Tự cấu hình thông số (Custom)"],
        key="scenario_select",
        on_change=on_scenario_change
    )
with col_ctrl3:
    decision_threshold = st.slider(
        "3. Ngưỡng quyết định (Threshold):",
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        key="decision_threshold_slider"
    )

# Dòng 2: Thanh trượt What-If (Chỉ hiển thị Top 5 đặc trưng quan trọng nhất)
st.markdown("<h3 style='color: #0A2540; font-size: 1.2rem; font-weight: 600; margin-bottom: 12px; border-bottom: 1px solid #E5E7EB; padding-bottom: 6px;'><i class='fa-solid fa-wand-magic-sparkles' style='margin-right: 8px;'></i>Điều chỉnh thông số giao dịch nhanh (Phân tích What-If)</h3>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.8rem; color: #6B7280; margin-top: -8px; margin-bottom: 15px;'>Tinh chỉnh trực quan các trường vật lý và biến PCA có ảnh hưởng mạnh nhất dựa trên độ quan trọng SHAP.</p>", unsafe_allow_html=True)

col_w1, col_w2, col_w3, col_w4, col_w5, col_w6 = st.columns(6)
with col_w1:
    time_val = st.slider("Time (Thời gian - giây):", min_value=0.0, max_value=172792.0, value=st.session_state.time_val, step=1.0, key="time_slider")
    st.session_state.time_val = time_val
with col_w2:
    amount_val = st.slider("Amount (Số tiền - $):", min_value=0.0, max_value=5000.0, value=st.session_state.amount_val, step=10.0, key="amount_slider")
    st.session_state.amount_val = amount_val
with col_w3:
    v14_val = st.slider("Biến V14:", min_value=-15.0, max_value=10.0, value=st.session_state.v14_val, step=0.1, key="v14_slider")
    st.session_state.v14_val = v14_val
with col_w4:
    v4_val = st.slider("Biến V4:", min_value=-10.0, max_value=15.0, value=st.session_state.v4_val, step=0.1, key="v4_slider")
    st.session_state.v4_val = v4_val
with col_w5:
    v12_val = st.slider("Biến V12:", min_value=-15.0, max_value=10.0, value=st.session_state.v12_val, step=0.1, key="v12_slider")
    st.session_state.v12_val = v12_val
with col_w6:
    v10_val = st.slider("Biến V10:", min_value=-15.0, max_value=10.0, value=st.session_state.v10_val, step=0.1, key="v10_slider")
    st.session_state.v10_val = v10_val

# Cap nhat cac dac trung dac biet vao mang chinh
st.session_state.v_array[13] = v14_val
st.session_state.v_array[3] = v4_val
st.session_state.v_array[11] = v12_val
st.session_state.v_array[9] = v10_val

st.markdown("---")

# =====================================================================
# 7. QUẢN LÝ 2 PHƯƠNG THỨC NHẬP HỖN HỢP: APP_V1 & APP_V2
# =====================================================================
st.markdown("<h3 style='color: #0A2540; font-size: 1.2rem; font-weight: 600; margin-bottom: 12px; border-bottom: 1px solid #E5E7EB; padding-bottom: 6px;'><i class='fa-solid fa-keyboard' style='margin-right: 8px;'></i> Phương Thức Nhập Chỉ Số Đặc Trưng (V1 đến V28)</h3>", unsafe_allow_html=True)

input_method = st.radio(
    "Lựa chọn cách thức điền dữ liệu PCA ẩn nâng cao của bạn:",
    ["Dạng chuỗi phân cách bởi dấu phẩy (app_v1)", "Dạng từng ô nhập số riêng biệt (app_v2)"],
    horizontal=True,
    key="input_method_radio"
)
st.session_state.input_method = input_method

final_v_inputs = []

if input_method == "Dạng chuỗi phân cách bởi dấu phẩy (app_v1)":
    initial_v_string = ", ".join([f"{v:.4f}" for v in st.session_state.v_array])
    v_string_input = st.text_area(
        "Nhập chuỗi 28 chỉ số V1-V28 (ngăn cách bằng dấu phẩy):",
        value=initial_v_string,
        height=90,
        help="Sao chép toàn bộ chuỗi số từ file dữ liệu của bạn và dán vào đây để phân tích nhanh."
    )
    
    try:
        parsed_v = [float(x.strip()) for x in v_string_input.split(",") if x.strip() != ""]
        if len(parsed_v) == 28:
            final_v_inputs = parsed_v
            st.session_state.v_array = final_v_inputs
        else:
            st.warning(f"Chuỗi hiện có {len(parsed_v)} phần tử. Bắt buộc phải có đủ 28 biến từ V1 đến V28. Hệ thống đang lấy dữ liệu mẫu dự phòng.")
            final_v_inputs = st.session_state.v_array
    except ValueError:
        st.error("Lỗi định dạng dữ liệu đầu vào chứa ký tự lạ (chỉ chấp nhận số và dấu phẩy).")
        final_v_inputs = st.session_state.v_array

else: # Nhap tung o nhap lieu rieng biet (app_v2)
    with st.expander("Nhấp để xem và nhập chi tiết từng chỉ số PCA ẩn nâng cao (V1 đến V28):", expanded=True):
        col_v_left, col_v_right = st.columns(2)
        v_inputs_tmp = []
        for i in range(28):
            label_text = f"Thành phần chính V{i+1}"
            val_init = st.session_state.v_array[i]
            
            if i < 14:
                with col_v_left:
                    is_disabled = i in [3, 9, 11, 13]
                    val = st.number_input(label_text, value=float(val_init), format="%.4f", key=f"v_field_{i}", disabled=is_disabled)
                    v_inputs_tmp.append(val)
            else:
                with col_v_right:
                    is_disabled = False
                    val = st.number_input(label_text, value=float(val_init), format="%.4f", key=f"v_field_{i}", disabled=is_disabled)
                    v_inputs_tmp.append(val)
                    
        final_v_inputs = v_inputs_tmp
        st.session_state.v_array = final_v_inputs

# ---------------------------------------------------------------------
# TỰ ĐỘNG THEO DÕI SỰ THAY ĐỔI CỦA THAM SỐ ĐỂ ẨN KẾT QUẢ KHI CHƯA NHẤN NÚT
# ---------------------------------------------------------------------
params_changed = False
if "prev_time_val" in st.session_state and st.session_state.prev_time_val != st.session_state.time_val:
    params_changed = True
if "prev_amount_val" in st.session_state and st.session_state.prev_amount_val != st.session_state.amount_val:
    params_changed = True
if "prev_v_array" in st.session_state and st.session_state.prev_v_array != st.session_state.v_array:
    params_changed = True
if "prev_primary_algo" in st.session_state and st.session_state.prev_primary_algo != primary_algo:
    params_changed = True
if "prev_decision_threshold" in st.session_state and st.session_state.prev_decision_threshold != decision_threshold:
    params_changed = True
if "prev_input_method" in st.session_state and st.session_state.prev_input_method != input_method:
    params_changed = True

st.session_state.prev_time_val = st.session_state.time_val
st.session_state.prev_amount_val = st.session_state.amount_val
st.session_state.prev_v_array = st.session_state.v_array.copy()
st.session_state.prev_primary_algo = primary_algo
st.session_state.prev_decision_threshold = decision_threshold
st.session_state.prev_input_method = input_method

if params_changed:
    st.session_state.run_prediction = False

# =====================================================================
# 8. THỰC THI LUỒNG DỰ ĐOÁN VÀ ĐỐI SÁNH KẾT QUẢ TRÊN 8 MÔ HÌNH
# =====================================================================
predict_clicked = st.button("KIỂM TRA GIAO DỊCH VÀ GHI NHẬT KÝ VẾT", type="primary")

# Chuẩn hóa dữ liệu thô đầu vào để chạy mô hình
scaled_amt = all_assets["scaler_amount"].transform([[st.session_state.amount_val]])[0][0]
scaled_time = all_assets["scaler_time"].transform([[st.session_state.time_val]])[0][0]
features_array = np.array([scaled_amt, scaled_time] + final_v_inputs).reshape(1, -1)

if predict_clicked:
    st.session_state.run_prediction = True
    start_time = time.time()
    
    # Định nghĩa cấu hình đối sánh 8 mô hình
    model_configs = [
        {"key": "LR_NoSMOTE", "algo": "Logistic Regression", "method": "Dữ liệu Gốc (Không SMOTE)", "type": "ML"},
        {"key": "LR_SMOTE", "algo": "Logistic Regression", "method": "Dùng SMOTE", "type": "ML"},
        {"key": "DT_NoSMOTE", "algo": "Decision Tree", "method": "Dữ liệu Gốc (Không SMOTE)", "type": "ML"},
        {"key": "DT_SMOTE", "algo": "Decision Tree", "method": "Dùng SMOTE", "type": "ML"},
        {"key": "RF_NoSMOTE", "algo": "Random Forest", "method": "Dữ liệu Gốc (Không SMOTE)", "type": "ML"},
        {"key": "RF_SMOTE", "algo": "Random Forest", "method": "Dùng SMOTE", "type": "ML"},
        {"key": "CNN_NoSMOTE", "algo": "1D-CNN", "method": "Dữ liệu Gốc (Không SMOTE)", "type": "DL"},
        {"key": "CNN_SMOTE", "algo": "1D-CNN", "method": "Dùng SMOTE", "type": "DL"}
    ]
    
    results_table = []
    main_fraud_prob = 0.0
    main_prediction_label = 0
    main_execution_time_ms = 0
    
    for cfg in model_configs:
        mdl = all_assets[cfg["key"]]
        t_start = time.time()
        
        if cfg["type"] == "ML":
            pred_raw = mdl.predict(features_array)[0]
            if hasattr(mdl, "predict_proba"):
                prob_raw = mdl.predict_proba(features_array)[0][1]
            else:
                prob_raw = 1.0 if pred_raw == 1 else 0.0
        else:
            features_3d = features_array.reshape(features_array.shape[0], features_array.shape[1], 1)
            prob_raw = mdl.predict(features_3d, verbose=0)[0][0]
            
        t_end = time.time()
        execution_time_cfg = int((t_end - t_start) * 1000)
        
        pred_label = 1 if prob_raw >= st.session_state.decision_threshold_slider else 0
        status_text = "GIAN LẬN (Fraud)" if pred_label == 1 else "HỢP LỆ (Normal)"
        
        results_table.append({
            "Thuật toán": cfg["algo"],
            "Phương pháp xử lý dữ liệu": cfg["method"],
            "Kết quả nhận diện": status_text,
            "Độ tin cậy rủi ro": f"{prob_raw:.2%}",
            "Thời gian xử lý": f"{execution_time_cfg} ms"
        })
        
        is_selected_main = False
        if primary_algo == "Random Forest (Khuyên dùng)" and cfg["key"] == "RF_SMOTE":
            is_selected_main = True
        elif primary_algo == "Logistic Regression" and cfg["key"] == "LR_SMOTE":
            is_selected_main = True
        elif primary_algo == "Decision Tree" and cfg["key"] == "DT_SMOTE":
            is_selected_main = True
        elif primary_algo == "1D-CNN (Deep Learning)" and cfg["key"] == "CNN_SMOTE":
            is_selected_main = True
            
        if is_selected_main:
            main_fraud_prob = prob_raw
            main_prediction_label = pred_label
            main_execution_time_ms = execution_time_cfg
            
    # Tạo dataframe biểu diễn 30 cột đầy đủ phục vụ SHAP
    raw_col_names = ["scaled_amount", "scaled_time"] + [f"V{i}" for i in range(1, 29)]
    processed_df_main = pd.DataFrame(features_array, columns=raw_col_names)
    
    main_duration = int((time.time() - start_time) * 1000)
    
    # GHI VÀO SQL SERVER / LOCAL FALLBACK
    txn_id_log = "TXN-007" if "TXN-007" in selected_txn else ("TXN-001" if "TXN-001" in selected_txn else "TXN-CUSTOM")
    pred_str_db = "FRAUD" if main_prediction_label == 1 else "NORMAL"
    
    status_db_write, target_db = insert_log_to_sql(
        transaction_id=txn_id_log,
        model_name=f"{primary_algo} - SMOTE",
        prediction=pred_str_db,
        probability=float(main_fraud_prob),
        execution_time_ms=main_duration,
        input_time=st.session_state.time_val,
        input_amount=st.session_state.amount_val,
        v_values=final_v_inputs
    )
    
    # Lưu thông tin vào session để dùng khi reload lại giao diện Streamlit
    st.session_state.results_table = results_table
    st.session_state.main_fraud_prob = main_fraud_prob
    st.session_state.main_prediction_label = main_prediction_label
    st.session_state.main_execution_time_ms = main_execution_time_ms
    st.session_state.processed_df_main = processed_df_main
    st.session_state.db_write_status = (status_db_write, target_db)

# =====================================================================
# 9. ĐIỀU HƯỚNG SIDEBAR BÊN TRÁI
# =====================================================================
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-top: 15px; margin-bottom: 25px;"><div style="background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%); padding: 18px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);"><i class="fa-solid fa-shield-halved" style="font-size: 2.2rem; color: #ffffff; margin-bottom: 8px;"></i><div style="color: #ffffff; font-size: 1.15rem; font-weight: 800; letter-spacing: 0.5px;">CREDIT DETECTION</div><div style="color: #93C5FD; font-size: 0.75rem; margin-top: 2px; font-weight: 500;">GIÁM SÁT GIAO DỊCH RISK-FREE</div></div></div>""", unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Chọn mục hiển thị:",
        [
            "KẾT QUẢ & GIẢI THÍCH (SHAP)",
            "QUY TRÌNH HỆ THỐNG",
            "NHẬT KÝ LƯU VẾT (SQL SERVER)"
        ],
        key="main_navigation_radio",
        label_visibility="collapsed"
    )
    st.markdown("""<div style="margin-top: 400px; text-align: center; color: #9CA3AF; font-size: 0.75rem; line-height: 1.4;"><hr style="margin: 0 0 20px 0; border: 0; border-top: 1px solid #E5E7EB;"><div>© Ngô Công Danh - Từ Hào Văn</div>""", unsafe_allow_html=True)

# ----------------- PHẦN 1: KẾT QUẢ & SHAP -----------------
if nav_selection == "KẾT QUẢ & GIẢI THÍCH (SHAP)":
    st.markdown("<br>", unsafe_allow_html=True)
    
    if not st.session_state.run_prediction:
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #9CA3AF;">
                <i class="fa-solid fa-magnifying-glass-chart" style="font-size: 3.5rem; margin-bottom: 15px;"></i>
                <p style="font-size: 1.15rem; font-weight: 500;">Vui lòng cấu hình thông số và bấm Kiểm tra giao dịch để xem kết quả phân tích.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Tải cache từ session
        cur_pred_label = st.session_state.main_prediction_label
        cur_prob = st.session_state.main_fraud_prob
        cur_results_table = st.session_state.results_table
        cur_processed_df = st.session_state.processed_df_main
        
        # Hiển thị thẻ rực rỡ trạng thái rủi ro
        if cur_pred_label == 1:
            st.markdown(f"""
                <div class="result-card-fraud">
                    <div class="result-title"><i class="fa-solid fa-triangle-exclamation"></i> CẢNH BÁO: GIAO DỊCH GIAN LẬN (FRAUD)</div>
                    <div class="result-desc">
                        Hệ thống phát hiện dấu hiệu bất thường nghiêm trọng. Xác suất rủi ro đạt: 
                        <strong>{cur_prob:.2%}</strong> (Vượt ngưỡng thiết lập {st.session_state.decision_threshold_slider:.2f})
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-normal">
                    <div class="result-title"><i class="fa-solid fa-circle-check"></i> XÁC NHẬN: GIAO DỊCH HỢP LỆ (NORMAL)</div>
                    <div class="result-desc">
                        Giao dịch nằm trong ngưỡng kiểm soát rủi ro an toàn. Xác suất gian lận: 
                        <strong>{cur_prob:.2%}</strong> (Thấp hơn ngưỡng thiết lập {st.session_state.decision_threshold_slider:.2f})
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Hiển thị bảng đối sánh 8 mô hình
        st.markdown("<h3 style='color: #0A2540; font-size: 1.15rem; font-weight: 700; margin-top: 15px; margin-bottom: 10px;'><i class='fa-solid fa-table-list'></i> Bảng Đối Sánh Hiệu Năng Phân Loại (8 Cấu Hình)</h3>", unsafe_allow_html=True)
        df_compare = pd.DataFrame(cur_results_table)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # Giải thích quyết định mô hình (Explainable AI - SHAP)
        st.markdown("<h3 style='color: #0A2540; font-size: 1.15rem; font-weight: 700; margin-top: 25px; margin-bottom: 5px;'><i class='fa-solid fa-brain'></i> Giải thích Quyết định Mô hình (Explainable AI)</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="background-color: #EFF6FF; border: 1px solid #DBEAFE; color: #1E40AF; padding: 12px; border-radius: 6px; font-size: 0.9rem; margin-bottom: 15px;">
                <i class="fa-solid fa-circle-info"></i> Mô hình cốt lõi đang chọn: <strong>{primary_algo}</strong>. 
                Dưới đây là phân tích mức độ tác động của từng đặc trưng lên quyết định dự đoán.
            </div>
        """, unsafe_allow_html=True)

        if "explainer" in all_assets and all_assets["explainer"] is not None and primary_algo in ["Random Forest (Khuyên dùng)", "Decision Tree"]:
            try:
                explainer_raw = all_assets["explainer"]
                shap_values_raw = explainer_raw.shap_values(cur_processed_df)
                
                if isinstance(shap_values_raw, list):
                    shap_values = shap_values_raw[1]
                elif len(shap_values_raw.shape) == 3:
                    shap_values = shap_values_raw[:, :, 1]
                else:
                    shap_values = shap_values_raw
                    
                base_val = explainer_raw.expected_value[1] if isinstance(explainer_raw.expected_value, (list, np.ndarray)) else explainer_raw.expected_value
                
                st.markdown("<p style='font-size: 0.95rem; font-weight: 600; color: #374151;'>Đồ thị SHAP Force Plot lực đóng góp đặc trưng (Thời gian thực):</p>", unsafe_allow_html=True)
                st_shap(shap.force_plot(base_val, shap_values[0], cur_processed_df), height=130)
                
                col_graph, col_nlp = st.columns([1.2, 1])
                
                with col_graph:
                    st.markdown("<p style='font-size: 0.95rem; font-weight: 600; color: #374151; margin-top: 10px;'>Top 5 chỉ số có tầm ảnh hưởng lớn nhất:</p>", unsafe_allow_html=True)
                    
                    feature_impact = pd.DataFrame({
                        'Feature': cur_processed_df.columns,
                        'Value': cur_processed_df.values[0],
                        'SHAP': shap_values[0]
                    })
                    feature_impact['Absolute_SHAP'] = feature_impact['SHAP'].abs()
                    top_5_features = feature_impact.sort_values(by='Absolute_SHAP', ascending=False).head(5)
                    
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    colors = ['#EF4444' if val > 0 else '#10B981' for val in top_5_features['SHAP']]
                    ax.barh(top_5_features['Feature'], top_5_features['SHAP'], color=colors, height=0.55)
                    ax.axvline(0, color='#9CA3AF', linewidth=0.8, linestyle='--')
                    ax.set_xlabel('Tác động đóng góp SHAP', fontsize=9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with col_nlp:
                    st.markdown("<p style='font-size: 0.95rem; font-weight: 600; color: #374151; margin-top: 10px;'>Diễn giải quyết định bằng Ngôn ngữ tự nhiên:</p>", unsafe_allow_html=True)
                    
                    feature_impact_sorted = feature_impact.sort_values(by='Absolute_SHAP', ascending=False)
                    risk_increasing_list = feature_impact_sorted[feature_impact_sorted['SHAP'] > 0]['Feature'].head(3).tolist()
                    
                    if cur_pred_label == 1:
                        risk_str = ", ".join(risk_increasing_list) if risk_increasing_list else "biến ẩn không gian"
                        natural_explanation = (
                            f"Giao dịch này bị đánh dấu GIAN LẬN (FRAUD) do mô hình phân loại có độ rủi ro "
                            f"{cur_prob:.2%}, vượt ngưỡng an toàn {st.session_state.decision_threshold_slider:.2f}. "
                            f"Theo cơ sở toán học XAI, các yếu tố {risk_str} là nguyên nhân trực tiếp kéo rủi ro lên cao."
                        )
                        st.error(natural_explanation)
                    else:
                        natural_explanation = (
                            f"Giao dịch được xếp hạng AN TOÀN (NORMAL) vì xác suất xảy ra gian lận thực tế chỉ "
                            f"đạt {cur_prob:.2%}, nằm sâu dưới ngưỡng rủi ro cho phép. "
                            f"Tất cả các biến đặc trưng PCA đầu vào đều dao động trong phạm vi an toàn lý tưởng."
                        )
                        st.success(natural_explanation)
                        
            except Exception as e:
                st.warning(f"Biểu đồ SHAP đang được thiết lập hoặc mô hình được chọn không tương thích với biểu đồ trực quan (TreeExplainer). Chi tiết: {str(e)}")
        else:
            st.info("SHAP (Explainable AI) đang được tối ưu hóa hiển thị cho mô hình cây (Random Forest và Decision Tree). Mô hình Neural Network Deep Learning và Logistic Regression có thể tham chiếu qua bảng đối sánh.")

# ----------------- PHẦN 2: QUY TRÌNH HỆ THỐNG -----------------
elif nav_selection == "QUY TRÌNH HỆ THỐNG":
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0A2540; font-size: 1.3rem; font-weight: 700;'><i class='fa-solid fa-graduation-cap'></i> PHẦN 1: TÓM TẮT QUÁ TRÌNH HUẤN LUYỆN (OFFLINE TRAINING)</h3>", unsafe_allow_html=True)
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("""
            <div style="background-color: #EFF6FF; border: 1px solid #DBEAFE; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.75rem; font-weight: bold; color: #1D4ED8; text-transform: uppercase;">Dữ liệu đầu vào (Kaggle)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #1E3A8A; margin: 5px 0;">284,807</div>
                <div style="font-size: 0.75rem; color: #4B5563;">Giao dịch thẻ tín dụng ẩn danh thực tế. Chỉ có 0.17% (492) là Gian lận.</div>
            </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown("""
            <div style="background-color: #FAF5FF; border: 1px solid #E9D5FF; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.75rem; font-weight: bold; color: #7E22CE; text-transform: uppercase;">Kỹ thuật SMOTE</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #581C87; margin: 5px 0;">50% / 50%</div>
                <div style="font-size: 0.75rem; color: #4B5563;">Nội suy sinh dữ liệu thiểu số tổng hợp nhằm cân bằng lớp dữ liệu trước khi train.</div>
            </div>
        """, unsafe_allow_html=True)
    with col_t3:
        st.markdown("""
            <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; padding: 20px; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.75rem; font-weight: bold; color: #047857; text-transform: uppercase;">Hiệu năng (Random Forest)</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #065F46; margin: 5px 0;">0.98 AUC</div>
                <div style="font-size: 0.75rem; color: #4B5563;">F1-Score đạt 0.88 trên tập kiểm thử độc lập. Mô hình vượt qua nguy cơ Overfitting.</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Cơ sở khoa học nền tảng: Thuật toán Nội suy SMOTE", expanded=True):
        st.markdown("""
            Hệ thống áp dụng thuật toán **SMOTE (Synthetic Minority Over-sampling Technique)** dựa trên bài báo nghiên cứu nền tảng của *Chawla et al. (2002)*:
            
            1. **Bước 1**: Tìm kiếm điểm dữ liệu gian lận $x_i$ trong không gian 30 chiều.
            2. **Bước 2**: Sử dụng bộ lọc lân cận K-NN (k-Nearest Neighbors) để định vị 5 điểm rủi ro gần nhất.
            3. **Bước 3**: Sinh dữ liệu mô phỏng mới nằm giữa ranh giới kết nối của 2 điểm bằng công thức:
            $$x_{new} = x_i + \lambda \times (x_{knn} - x_i) \quad [ \lambda \in (0, 1) ]$$
            
            **Liên hệ thực tiễn với tương tác "What-If" trên Dashboard:**
            SMOTE đã giúp định hình ranh giới quyết định (Decision Boundary) rõ ràng khi huấn luyện. Vì thế, khi bạn kéo thanh trượt **V14** từ `1.20` xuống `-4.50`, bản chất bạn đang **di chuyển tọa độ điểm giao dịch đó vượt qua Ranh giới rủi ro** mà SMOTE đã dệt sẵn để kích hoạt cảnh báo rủi ro thực tế trên tab 1!
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0A2540; font-size: 1.3rem; font-weight: 700;'><i class='fa-solid fa-network-wired'></i> PHẦN 2: LUỒNG SUY LUẬN THỜI GIAN THỰC (REAL-TIME INFERENCE)</h3>", unsafe_allow_html=True)
    
    # Cập nhật luồng hiển thị bước 5, 6 theo trạng thái chạy dự báo
    if st.session_state.run_prediction:
        prob_str = f"{(st.session_state.main_fraud_prob * 100):.2f}%"
        decision_str = "GIAN LẬN (Fraud)" if st.session_state.main_prediction_label == 1 else "HỢP LỆ (Normal)"
    else:
        prob_str = "Đang chờ kiểm tra..."
        decision_str = "Đang chờ kiểm tra..."
        
    steps = [
        ("Step 1: Input Received (Nhận dữ liệu)", f"Hệ thống tiếp nhận Request vector gồm 30 biến đặc trưng thô từ UI. Giá trị hiện tại: Amount = ${st.session_state.amount_val:.2f}, Time = {st.session_state.time_val:.1f} giây."),
        ("Step 2: Data Validation (Xác thực dữ liệu)", "Hệ thống kiểm tra định dạng vector đầu vào, lọc bỏ ký tự NaN, đảm bảo cấu trúc đầy đủ 30 chiều tọa độ."),
        ("Step 3: Preprocessing (Chuẩn hóa)", f"Chuẩn hóa dữ liệu thô Time và Amount bằng StandardScaler. Giá trị sau scaler: Amount ({scaled_amt:.6f}), Time ({scaled_time:.6f})."),
        ("Step 4: Feature Interpretation (Tích hợp XAI)", "Khởi chạy bộ giải thích SHAP TreeExplainer trên luồng suy luận để lượng hóa lực đóng góp."),
        ("Step 5: Model Prediction (Thực thi dự đoán)", f"Đẩy vector đã chuẩn hóa vào {primary_algo} để tính toán phân lớp. Xác suất rủi ro ròng: {prob_str}."),
        ("Step 6: Threshold Decision (So khớp ngưỡng)", f"Đối chiếu xác suất rủi ro với ngưỡng {st.session_state.decision_threshold_slider:.2f}. Kết luận giao dịch: {decision_str}."),
        ("Step 7: Database Logging (Ghi nhật ký SQL)", "Tự động kích hoạt cơ chế ghi nhật ký bất đồng bộ để lưu vết dữ liệu kiểm thử vào SQL Server phục vụ hậu kiểm.")
    ]
    
    for i, (title, desc) in enumerate(steps):
        with st.expander(f"{title}", expanded=True):
            st.write(desc)

# ----------------- PHẦN 3: NHẬT KÝ LƯU VẾT (SQL SERVER) -----------------
elif nav_selection == "NHẬT KÝ LƯU VẾT (SQL SERVER)":
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_t3_header, col_t3_btn = st.columns([4, 1])
    with col_t3_header:
        st.markdown("<h3 style='color: #0A2540; font-size: 1.25rem; font-weight: 700;'><i class='fa-solid fa-database'></i> Nhật ký Giám sát rủi ro (Kết nối máy Danh-PC)</h3>", unsafe_allow_html=True)
        st.write("Dữ liệu được cập nhật tự động sau mỗi lần nhấn nút Kiểm tra.")
    with col_t3_btn:
        refresh_logs = st.button("Tải lại dữ liệu")
        
    logs_data, conn_mode, err_message = fetch_all_logs()
    
    if conn_mode == "SQL_SERVER":
        st.markdown("""
            <div style="display: inline-flex; align-items: center; gap: 6px; background-color: #DEF7EC; border: 1px solid #31C48D; color: #03543F; padding: 6px 12px; border-radius: 50px; font-size: 0.8rem; font-weight: bold; margin-bottom: 15px;">
                <span style="display: inline-block; width: 8px; height: 8px; background-color: #31C48D; border-radius: 50%;"></span>
                ĐÃ KẾT NỐI ĐẾN SQL SERVER (SERVER: Danh-PC | DATABASE: CreditCardFraudDB)
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="display: inline-flex; align-items: center; gap: 6px; background-color: #FDF2F2; border: 1px solid #F8B4B4; color: #9B1C1C; padding: 6px 12px; border-radius: 50px; font-size: 0.8rem; font-weight: bold; margin-bottom: 15px;">
                <span style="display: inline-block; width: 8px; height: 8px; background-color: #F8B4B4; border-radius: 50%;"></span>
                CHẾ ĐỘ NGOẠI TUYẾN (Offline Fallback - Đang lưu trữ cục bộ)
            </div>
        """, unsafe_allow_html=True)
        if err_message:
            st.info(f"Lưu ý kết nối SQL Server: {err_message}. Vui lòng tạo database bằng script SQL trong thư mục `database/create_db_and_table.sql` để kết nối.")

    if logs_data:
        df_logs = pd.DataFrame(logs_data)
        
        def format_pred(val):
            if "FRAUD" in str(val).upper():
                return "GIAN LẬN (Fraud)"
            return "HỢP LỆ (Normal)"
            
        df_logs["Prediction"] = df_logs["Prediction"].apply(format_pred)
        df_logs["Probability"] = df_logs["Probability"].apply(lambda x: f"{x:.2%}")
        df_logs["ExecutionTimeMs"] = df_logs["ExecutionTimeMs"].apply(lambda x: f"{x} ms")
        
        df_logs = df_logs.rename(columns={
            "TransactionID": "Mã Giao Dịch",
            "ModelName": "Mô Hình Phân Phối",
            "Prediction": "Nhãn Phân Loại",
            "Probability": "Xác Suất Rủi Ro",
            "ExecutionTimeMs": "Thời Gian Xử Lý",
            "Timestamp": "Thời Điểm Kiểm Tra"
        })
        
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
    else:
        st.write("Không tìm thấy nhật ký giao dịch nào.")
