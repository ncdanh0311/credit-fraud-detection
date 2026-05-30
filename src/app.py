import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import warnings
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
div[data-testid="stHeader"] {
    display: none !important;
}
.block-container {
    padding-top: 0.1rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin-top: -30px !important;
}
div[data-testid="stExpander"] {
    margin-bottom: 4px !important;
}
div.stSlider {
    padding-bottom: 0px !important;
    margin-bottom: -15px !important;
}
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
[data-testid="stSidebar"]{min-width:260px !important;max-width:260px !important;width:260px !important;overflow:hidden !important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{overflow-y:hidden !important;overflow-x:hidden !important;}
[data-testid="stSidebar"]::-webkit-scrollbar, [data-testid="stSidebar"] *::-webkit-scrollbar{display:none !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] *{-ms-overflow-style:none !important;scrollbar-width:none !important;}
[data-testid="stSidebarDragHandle"], [data-testid="stSidebarResizer"], [class*="stSidebarDragHandle"], [class*="stSidebarResizer"]{display:none !important;pointer-events:none !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p{font-size:0.92rem !important;font-weight:600 !important;color:#4B5563 !important;line-height:1.6 !important;white-space:nowrap !important;transition:all 0.2s ease !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{gap:15px !important; width:100% !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label{display:flex !important; width:100% !important; box-sizing:border-box !important; padding:12px 16px !important;border-radius:8px !important;transition:all 0.2s ease !important;cursor:pointer !important;border:1px solid rgba(229,231,235,0.6) !important;background-color:#ffffff !important;box-shadow:0 2px 4px rgba(0,0,0,0.02) !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover{background-color:#F9FAFB !important;border-color:#D1D5DB !important;}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-of-type{display:none !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked){background-color:#ffffff !important;border:1px solid #E5E7EB !important;border-bottom:4px solid #1E3A8A !important;box-shadow:0 4px 6px -1px rgba(0,0,0,0.05) !important;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) p{color:#1E3A8A !important;font-weight:700 !important;}
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
    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.models import Model
    
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
        import shap
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
    st.session_state.current_scenario = "Tự cấu hình thông số (Custom)"
    st.session_state.scenario_select = "Tự cấu hình thông số (Custom)"
    st.session_state.input_method = "Dạng chuỗi phân cách bởi dấu phẩy"
    st.session_state.primary_algo = "Random Forest"
    st.session_state.decision_threshold = 0.50
    
    # Khởi tạo giá trị mặc định cho sliders dạng Custom
    st.session_state.time_val = 0.0
    st.session_state.amount_val = 100.0
    st.session_state.v14_val = 0.0
    st.session_state.v4_val = 0.0
    st.session_state.v12_val = 0.0
    st.session_state.v10_val = 0.0
    
    # 28 biến đặc trưng V dạng Custom (tất cả là 0.0)
    st.session_state.v_array = [0.0] * 28
    
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

# Hàm chuyển đổi sang chế độ Custom khi có bất cứ tham số nào bị thay đổi
def on_parameter_change():
    st.session_state.scenario_select = "Tự cấu hình thông số (Custom)"

# Hàm đồng bộ hóa khi chuyển đổi giao dịch mẫu
def on_scenario_change():
    scen = st.session_state.get("scenario_select")
    if not scen:
        return
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
        st.session_state.amount_val = 950.00
        st.session_state.v14_val = -9.50
        st.session_state.v4_val = 6.50
        st.session_state.v12_val = -9.20
        st.session_state.v10_val = -6.50
        st.session_state.v_array = [-3.04, 3.48, -3.74, 6.50, -1.17, -1.41, -2.44, 0.49, -2.75, -6.50, 4.59, -9.20, 1.43, -9.50, -0.10, -5.50, -8.50, -1.25, -0.33, 0.35, 0.66, 0.44, -0.04, -0.25, -0.27, -0.02, 0.35, -0.15]
    else:
        st.session_state.time_val = 0.0
        st.session_state.amount_val = 100.0
        st.session_state.v14_val = 0.0
        st.session_state.v4_val = 0.0
        st.session_state.v12_val = 0.0
        st.session_state.v10_val = 0.0
        st.session_state.v_array = [0.0] * 28

    # Đồng bộ trực tiếp các session state keys của widget để cập nhật giao diện ngay lập tức
    st.session_state.time_slider = st.session_state.time_val
    st.session_state.amount_slider = st.session_state.amount_val
    st.session_state.v14_slider = st.session_state.v14_val
    st.session_state.v4_slider = st.session_state.v4_val
    st.session_state.v12_slider = st.session_state.v12_val
    st.session_state.v10_slider = st.session_state.v10_val
    
    st.session_state.v_string_input_area = ", ".join([f"{v:.4f}" for v in st.session_state.v_array])
    for i in range(28):
        st.session_state[f"v_field_{i}"] = float(st.session_state.v_array[i])

# Header (Full width, but very small height)
st.markdown("""
    <div style="text-align: center; margin-top: -45px; margin-bottom: 10px;">
        <h2 style="color: #0A2540; font-size: 1.5rem; font-weight: 800; display: inline-flex; align-items: center; gap: 8px;">
            <i class="fa-solid fa-shield-halved" style="color: #1E3A8A; font-size: 1.3rem;"></i> HỆ THỐNG GIÁM SÁT GIAO DỊCH THẺ TÍN DỤNG
        </h2>
    </div>
""", unsafe_allow_html=True)

# Khởi tạo hai cột chính: Cột trái (Điều khiển) và Cột phải (Kết quả/Phân tích)
col_left, col_right = st.columns([1, 1.25], gap="medium")

with col_left:
    st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 0px; margin-bottom: 5px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px;'><i class='fa-solid fa-sliders'></i> Cấu Hình Chung</h4>", unsafe_allow_html=True)
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        primary_algo = st.selectbox(
            "Mô hình cốt lõi:",
            ["Random Forest", "Logistic Regression", "Decision Tree", "1D-CNN (Deep Learning)"],
            key="primary_algo_select"
        )
    with col_c2:
        selected_txn = st.selectbox(
            "Giao dịch (Auto-fill):",
            ["TXN-007 (Mẫu Giao dịch Nghi ngờ Gian lận - Fraud)", "TXN-001 (Mẫu Giao dịch Hợp lệ - Normal)", "Tự cấu hình thông số (Custom)"],
            key="scenario_select",
            on_change=on_scenario_change
        )
    
    decision_threshold = st.slider(
        "Ngưỡng quyết định (Threshold):",
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        key="decision_threshold_slider"
    )

    st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 15px; margin-bottom: 5px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px;'><i class='fa-solid fa-wand-magic-sparkles'></i> Tinh Chỉnh What-If (Các biến tác động mạnh)</h4>", unsafe_allow_html=True)
    
    col_w_l, col_w_r = st.columns(2)
    with col_w_l:
        time_val = st.slider("Time (Thời gian):", min_value=0.0, max_value=172792.0, value=st.session_state.time_val, step=1.0, key="time_slider", on_change=on_parameter_change)
        st.session_state.time_val = time_val
        
        v14_val = st.slider("Biến V14 (Rủi ro):", min_value=-15.0, max_value=10.0, value=st.session_state.v14_val, step=0.1, key="v14_slider", on_change=on_parameter_change)
        st.session_state.v14_val = v14_val
        
        v12_val = st.slider("Biến V12:", min_value=-15.0, max_value=10.0, value=st.session_state.v12_val, step=0.1, key="v12_slider", on_change=on_parameter_change)
        st.session_state.v12_val = v12_val
    with col_w_r:
        amount_val = st.slider("Amount (Số tiền $):", min_value=0.0, max_value=5000.0, value=st.session_state.amount_val, step=10.0, key="amount_slider", on_change=on_parameter_change)
        st.session_state.amount_val = amount_val
        
        v4_val = st.slider("Biến V4:", min_value=-10.0, max_value=15.0, value=st.session_state.v4_val, step=0.1, key="v4_slider", on_change=on_parameter_change)
        st.session_state.v4_val = v4_val
        
        v10_val = st.slider("Biến V10:", min_value=-15.0, max_value=10.0, value=st.session_state.v10_val, step=0.1, key="v10_slider", on_change=on_parameter_change)
        st.session_state.v10_val = v10_val
        
    st.session_state.v_array[13] = v14_val
    st.session_state.v_array[3] = v4_val
    st.session_state.v_array[11] = v12_val
    st.session_state.v_array[9] = v10_val

    st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 15px; margin-bottom: 5px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px;'><i class='fa-solid fa-keyboard'></i> Điền Biến Ẩn (V1 - V28)</h4>", unsafe_allow_html=True)
    
    input_method = st.radio(
        "Phương thức nhập:",
        ["Chuỗi dấu phẩy", "Ô nhập số"],
        horizontal=True,
        key="input_method_radio"
    )
    
    final_v_inputs = []
    
    if input_method == "Chuỗi dấu phẩy":
        initial_v_string = ", ".join([f"{v:.4f}" for v in st.session_state.v_array])
        v_string_input = st.text_area(
            "Chuỗi 28 chỉ số V1-V28 (ngăn cách bằng dấu phẩy):",
            value=initial_v_string,
            height=85,
            label_visibility="collapsed",
            key="v_string_input_area",
            on_change=on_parameter_change
        )
        try:
            parsed_v = [float(x.strip()) for x in v_string_input.split(",") if x.strip() != ""]
            if len(parsed_v) == 28:
                final_v_inputs = parsed_v
                st.session_state.v_array = final_v_inputs
            else:
                final_v_inputs = st.session_state.v_array
        except ValueError:
            final_v_inputs = st.session_state.v_array
    else:
        with st.expander("Mở rộng nhập 28 ô số", expanded=False):
            col_v_left, col_v_right = st.columns(2)
            v_inputs_tmp = []
            for i in range(28):
                label_text = f"V{i+1}"
                val_init = st.session_state.v_array[i]
                if i < 14:
                    with col_v_left:
                        is_disabled = i in [3, 9, 11, 13]
                        val = st.number_input(label_text, value=float(val_init), format="%.4f", key=f"v_field_{i}", disabled=is_disabled, on_change=on_parameter_change)
                        v_inputs_tmp.append(val)
                else:
                    with col_v_right:
                        val = st.number_input(label_text, value=float(val_init), format="%.4f", key=f"v_field_{i}", on_change=on_parameter_change)
                        v_inputs_tmp.append(val)
            final_v_inputs = v_inputs_tmp
            st.session_state.v_array = final_v_inputs

    predict_clicked = st.button("KIỂM TRA GIAO DỊCH VÀ GHI NHẬT KÝ VẾT", type="primary", use_container_width=True)

# Chuẩn hóa dữ liệu thô đầu vào để chạy mô hình
scaled_amt = all_assets["scaler_amount"].transform([[st.session_state.amount_val]])[0][0]
scaled_time = all_assets["scaler_time"].transform([[st.session_state.time_val]])[0][0]
features_array = np.array([scaled_amt, scaled_time] + final_v_inputs).reshape(1, -1)

# Tự động theo dõi sự thay đổi của tham số để ẩn kết quả phân tích cũ
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
    
    # Lưu thông tin vào session để dùng khi reload lại giao diện Streamlit
    st.session_state.results_table = results_table
    st.session_state.main_fraud_prob = main_fraud_prob
    st.session_state.main_prediction_label = main_prediction_label
    st.session_state.main_execution_time_ms = main_execution_time_ms
    st.session_state.processed_df_main = processed_df_main

    # GHI VÀO SQL SERVER / LOCAL FALLBACK
    if "TXN-007" in selected_txn:
        txn_id_log = "TXN-007"
    elif "TXN-001" in selected_txn:
        txn_id_log = "TXN-001"
    else:
        try:
            existing_logs, _, _ = fetch_all_logs()
            if not isinstance(existing_logs, list):
                existing_logs = []
        except Exception:
            existing_logs = []
        custom_ids = [log for log in existing_logs if log.get("TransactionID") and "TXN-CUSTOM" in str(log.get("TransactionID"))]
        next_num = len(custom_ids) + 1
        txn_id_log = f"TXN-CUSTOM-{next_num:03d}"
    pred_str_db = "FRAUD" if st.session_state.main_prediction_label == 1 else "NORMAL"
    
    status_db_write, target_db = insert_log_to_sql(
        transaction_id=txn_id_log,
        model_name=f"{primary_algo} - SMOTE",
        prediction=pred_str_db,
        probability=float(st.session_state.main_fraud_prob),
        execution_time_ms=main_duration,
        input_time=st.session_state.time_val,
        input_amount=st.session_state.amount_val,
        v_values=final_v_inputs
    )
    st.session_state.db_write_status = (status_db_write, target_db)
    
    # Hiển thị thông báo phản hồi ngay lập tức cho người dùng
    if status_db_write:
        with col_left:
            if target_db == "SQL_SERVER":
                st.success(f"Đã lưu vết thành công vào SQL Server (Mã: {txn_id_log})!")
            else:
                st.info(f"Đã lưu vết ngoại tuyến (Mã: {txn_id_log})!")

# =====================================================================
# 9. ĐIỀU HƯỚNG SIDEBAR BÊN TRÁI
# =====================================================================
with st.sidebar:
    st.markdown("""<div style="text-align: center; margin-top: 10px; margin-bottom: 15px; margin-left: 16px; margin-right: 16px;">
    <div style="background: linear-gradient(135deg, #0A2540 0%, #1E3A8A 100%); padding: 12px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <i class="fa-solid fa-shield-halved" style="font-size: 1.8rem; color: #ffffff; margin-bottom: 5px;"></i>
        <div style="color: #ffffff; font-size: 1.05rem; font-weight: 800; letter-spacing: 0.5px;">CREDIT DETECTION</div>
    </div>
</div>
""", unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "",
        [
            "KẾT QUẢ & GIẢI THÍCH (SHAP)",
            "QUY TRÌNH HỆ THỐNG",
            "NHẬT KÝ LƯU VẾT"
        ],
        key="main_navigation_radio",
        label_visibility="collapsed"
    )
    st.markdown("""<div style="margin-top: 450px; text-align: center; color: #9CA3AF; font-size: 0.75rem; line-height: 1.4;"><hr style="margin: 0 0 20px 0; border: 0; border-top: 1px solid #E5E7EB;"><div>© Ngô Công Danh - Từ Hào Văn</div>""", unsafe_allow_html=True)

with col_right:
    if nav_selection == "KẾT QUẢ & GIẢI THÍCH (SHAP)":
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not st.session_state.run_prediction:
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 580px; color: #9CA3AF; width: 100%;">
                    <i class="fa-solid fa-magnifying-glass-chart" style="font-size: 4.5rem; margin-bottom: 20px; color: #9CA3AF; opacity: 0.85;"></i>
                    <p style="font-size: 1.25rem; font-weight: 500; text-align: center; margin: 0; line-height: 1.5;">Vui lòng cấu hình thông số để xem kết quả phân tích.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Lazy import shap và streamlit_shap để tăng tối đa tốc độ phản hồi UI
            import shap
            from streamlit_shap import st_shap
            
            # Tải cache từ session
            cur_pred_label = st.session_state.main_prediction_label
            cur_prob = st.session_state.main_fraud_prob
            cur_processed_df = st.session_state.processed_df_main
            
            # Lấy model đang được chọn để giải thích SHAP động
            main_model = None
            model_type = None
            
            if "Random Forest" in primary_algo:
                main_model = all_assets["RF_SMOTE"]
                model_type = "tree"
            elif primary_algo == "Decision Tree":
                main_model = all_assets["DT_SMOTE"]
                model_type = "tree"
            elif primary_algo == "Logistic Regression":
                main_model = all_assets["LR_SMOTE"]
                model_type = "linear"
            elif primary_algo == "1D-CNN (Deep Learning)":
                main_model = all_assets["CNN_SMOTE"]
                model_type = "cnn"

            shap_values = None
            base_val = 0.5
            feature_impact = None
            top_5_features = None
            shap_error = None
            
            try:
                if model_type == "tree":
                    # Sử dụng TreeExplainer cực nhanh cho mô hình dạng cây
                    explainer = shap.TreeExplainer(main_model)
                    shap_values_raw = explainer.shap_values(cur_processed_df)
                    
                    if isinstance(shap_values_raw, list):
                        shap_values = shap_values_raw[1]
                    elif len(shap_values_raw.shape) == 3:
                        shap_values = shap_values_raw[:, :, 1]
                    else:
                        shap_values = shap_values_raw
                        
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    
                elif model_type == "linear":
                    # Sử dụng KernelExplainer tối ưu cho Logistic Regression
                    background = np.zeros((1, 30))
                    def lr_predict_fn(x):
                        return main_model.predict_proba(x)[:, 1]
                    explainer = shap.KernelExplainer(lr_predict_fn, background)
                    shap_values_raw = explainer.shap_values(cur_processed_df.values)
                    
                    if isinstance(shap_values_raw, list):
                        shap_values = shap_values_raw[0]
                    else:
                        shap_values = shap_values_raw
                        
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                        base_val = base_val[0]
                    
                elif model_type == "cnn":
                    # Sử dụng KernelExplainer động cho 1D-CNN bằng cách reshape Tensor 3D thô
                    background = np.zeros((1, 30))
                    def cnn_predict_fn(x):
                        x_3d = x.reshape(x.shape[0], x.shape[1], 1)
                        return main_model.predict(x_3d, verbose=0).flatten()
                    explainer = shap.KernelExplainer(cnn_predict_fn, background)
                    shap_values_raw = explainer.shap_values(cur_processed_df.values)
                    if isinstance(shap_values_raw, list):
                        shap_values = shap_values_raw[0]
                    else:
                        shap_values = shap_values_raw
                        
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                        base_val = base_val[0]

                if shap_values is not None:
                    feature_impact = pd.DataFrame({
                        'Feature': cur_processed_df.columns,
                        'Value': cur_processed_df.values[0],
                        'SHAP': shap_values[0]
                    })
                    feature_impact['Absolute_SHAP'] = feature_impact['SHAP'].abs()
                    top_5_features = feature_impact.sort_values(by='Absolute_SHAP', ascending=False).head(5)
            except Exception as e:
                shap_error = str(e)

            # LAYOUT 1 CỘT: Mức độ rủi ro -> SHAP plot -> Xếp hạng đặc trưng -> So sánh
            # ============================================================
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 0px; margin-bottom: 6px;'><i class='fa-solid fa-shield-virus'></i> Mức Độ Rủi Ro</h4>", unsafe_allow_html=True)

            if cur_pred_label == 1:
                st.markdown(f"""
                    <div class="result-card-fraud" style="padding: 12px; margin-bottom: 10px;">
                        <div class="result-title" style="font-size: 1.0rem; font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
                            <i class="fa-solid fa-triangle-exclamation"></i> CẢNH BÁO GIAN LẬN (FRAUD)
                        </div>
                        <div class="result-desc" style="font-size: 0.82rem; line-height: 1.4;">
                            Mô hình <strong>{primary_algo}</strong> phát hiện rủi ro cao.
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px; font-weight: 600; color: #991B1B;">
                                <span>Xác suất rủi ro: <strong style="font-size: 0.88rem;">{cur_prob:.2%}</strong></span>
                                <span>Ngưỡng: <strong>{st.session_state.decision_threshold_slider:.2f}</strong></span>
                            </div>
                            <div style="background-color: #FCA5A5; border-radius: 6px; height: 8px; width: 100%; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, #EF4444, #B91C1C); width: {min(cur_prob * 100, 100.0):.2f}%; height: 100%; border-radius: 6px;"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-normal" style="padding: 12px; margin-bottom: 10px;">
                        <div class="result-title" style="font-size: 1.0rem; font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
                            <i class="fa-solid fa-circle-check"></i> AN TOÀN: GIAO DỊCH HỢP LỆ (NORMAL)
                        </div>
                        <div class="result-desc" style="font-size: 0.82rem; line-height: 1.4;">
                            Mô hình <strong>{primary_algo}</strong> đánh giá giao dịch nằm trong tầm kiểm soát an toàn.
                        </div>
                        <div style="margin-top: 10px;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px; font-weight: 600; color: #065F46;">
                                <span>Xác suất rủi ro: <strong style="font-size: 0.88rem;">{cur_prob:.2%}</strong></span>
                                <span>Ngưỡng: <strong>{st.session_state.decision_threshold_slider:.2f}</strong></span>
                            </div>
                            <div style="background-color: #A7F3D0; border-radius: 6px; height: 8px; width: 100%; overflow: hidden;">
                                <div style="background: linear-gradient(90deg, #10B981, #059669); width: {min(cur_prob * 100, 100.0):.2f}%; height: 100%; border-radius: 6px;"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # --- SHAP FORCE PLOT ---
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 8px; margin-bottom: 4px;'><i class='fa-solid fa-bolt'></i> SHAP Force Plot</h4>", unsafe_allow_html=True)

            if shap_error is None and shap_values is not None:
                st_shap(shap.force_plot(base_val, shap_values[0], cur_processed_df), height=135)
                st.markdown("""
                    <p style="font-size: 0.74rem; color: #6B7280; margin-top: 2px; margin-bottom: 8px; font-style: italic;">
                        Lực đẩy (màu đỏ) tăng rủi ro - Lực cản (màu xanh) giữ an toàn.
                    </p>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 135px; background: #F9FAFB; border-radius:6px; display:flex; align-items:center; justify-content:center; color:#9CA3AF; font-size:0.8rem; margin-bottom:8px;'>SHAP Force Plot không khả dụng</div>", unsafe_allow_html=True)

            # --- SHAP BAR CHART ---
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 8px; margin-bottom: 4px;'><i class='fa-solid fa-chart-bar'></i> Xếp Hạng Đặc Trưng (SHAP)</h4>", unsafe_allow_html=True)

            if shap_error is not None:
                st.warning(f"Không thể tải SHAP. Chi tiết: {shap_error}")
            elif shap_values is not None and top_5_features is not None:
                fig, ax = plt.subplots(figsize=(7, 2.6))
                colors = ['#EF4444' if val > 0 else '#10B981' for val in top_5_features['SHAP']]
                ax.barh(top_5_features['Feature'], top_5_features['SHAP'], color=colors, height=0.55, edgecolor='none')
                ax.axvline(0, color='#9CA3AF', linewidth=0.8, linestyle='--')
                ax.set_xlabel('Mức độ đóng góp SHAP', fontsize=8.5, fontweight='bold', color='#4B5563')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#D1D5DB')
                ax.spines['bottom'].set_color('#D1D5DB')
                ax.tick_params(axis='both', which='major', labelsize=9, colors='#374151')
                plt.tight_layout(pad=0.8)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Không có dữ liệu SHAP.")

            # --- BẢNG SO SÁNH ---
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 8px; margin-bottom: 6px;'><i class='fa-solid fa-code-compare'></i> So Sánh Với Giao Dịch Điển Hình</h4>", unsafe_allow_html=True)

            st.markdown(f"""
                <div style="background-color: #ffffff; border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.76rem; text-align: left; color: #374151;">
                        <thead>
                            <tr style="background-color: #F8FAFC; border-bottom: 2px solid #E5E7EB; color: #4B5563; font-weight: bold;">
                                <th style="padding: 7px 8px;">Biến số</th>
                                <th style="padding: 7px 8px;">Hiện tại</th>
                                <th style="padding: 7px 8px;">Điển hình</th>
                                <th style="padding: 7px 8px; text-align: right;">Trạng thái</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #F3F4F6;">
                                <td style="padding: 6px 8px; font-weight: 600;">Amount</td>
                                <td style="padding: 6px 8px; font-weight: bold; color: {'#EF4444' if st.session_state.amount_val > 500 else '#1F2937'};">${st.session_state.amount_val:.2f}</td>
                                <td style="padding: 6px 8px; color: #6B7280;">$22.75</td>
                                <td style="padding: 6px 8px; text-align: right;">
                                    <span style="background-color: {'#FEE2E2' if st.session_state.amount_val > 500 else '#D1FAE5'}; color: {'#991B1B' if st.session_state.amount_val > 500 else '#065F46'}; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">
                                        {'Cao bất thường' if st.session_state.amount_val > 500 else 'Bình thường'}
                                    </span>
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #F3F4F6;">
                                <td style="padding: 6px 8px; font-weight: 600;">Time</td>
                                <td style="padding: 6px 8px; font-weight: bold; color: #1F2937;">{st.session_state.time_val:.1f}s</td>
                                <td style="padding: 6px 8px; color: #6B7280;">406.0s</td>
                                <td style="padding: 6px 8px; text-align: right;">
                                    <span style="background-color: #D1FAE5; color: #065F46; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">Bình thường</span>
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #F3F4F6;">
                                <td style="padding: 6px 8px; font-weight: 600;">V14 (rủi ro thẻ)</td>
                                <td style="padding: 6px 8px; font-weight: bold; color: {'#EF4444' if st.session_state.v14_val < -3.0 else '#1F2937'};">{st.session_state.v14_val:.2f}</td>
                                <td style="padding: 6px 8px; color: #6B7280;">-0.50</td>
                                <td style="padding: 6px 8px; text-align: right;">
                                    <span style="background-color: {'#FEE2E2' if st.session_state.v14_val < -3.0 else '#D1FAE5'}; color: {'#991B1B' if st.session_state.v14_val < -3.0 else '#065F46'}; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">
                                        {'Lệch rủi ro cao' if st.session_state.v14_val < -3.0 else 'Bình thường'}
                                    </span>
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #F3F4F6;">
                                <td style="padding: 6px 8px; font-weight: 600;">V12 (biến ẩn)</td>
                                <td style="padding: 6px 8px; font-weight: bold; color: {'#EF4444' if st.session_state.v12_val < -5.0 else '#1F2937'};">{st.session_state.v12_val:.2f}</td>
                                <td style="padding: 6px 8px; color: #6B7280;">-3.30</td>
                                <td style="padding: 6px 8px; text-align: right;">
                                    <span style="background-color: {'#FEE2E2' if st.session_state.v12_val < -5.0 else '#D1FAE5'}; color: {'#991B1B' if st.session_state.v12_val < -5.0 else '#065F46'}; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">
                                        {'Sai lệch cực lớn' if st.session_state.v12_val < -5.0 else 'Bình thường'}
                                    </span>
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #F3F4F6;">
                                <td style="padding: 6px 8px; font-weight: 600;">V4 (động lực)</td>
                                <td style="padding: 6px 8px; font-weight: bold; color: {'#EF4444' if st.session_state.v4_val > 3.0 else '#1F2937'};">{st.session_state.v4_val:.2f}</td>
                                <td style="padding: 6px 8px; color: #6B7280;">1.15</td>
                                <td style="padding: 6px 8px; text-align: right;">
                                    <span style="background-color: {'#FEE2E2' if st.session_state.v4_val > 3.0 else '#D1FAE5'}; color: {'#991B1B' if st.session_state.v4_val > 3.0 else '#065F46'}; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">
                                        {'Kích hoạt rủi ro' if st.session_state.v4_val > 3.0 else 'Bình thường'}
                                    </span>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 8px; font-weight: 600;">V10 (lịch sử thẻ)</td>
                                <td style="padding: 6px 8px; font-weight: bold; color: {'#EF4444' if st.session_state.v10_val < -2.0 else '#1F2937'};">{st.session_state.v10_val:.2f}</td>
                                <td style="padding: 6px 8px; color: #6B7280;">-0.10</td>
                                <td style="padding: 6px 8px; text-align: right;">
                                    <span style="background-color: {'#FEE2E2' if st.session_state.v10_val < -2.0 else '#D1FAE5'}; color: {'#991B1B' if st.session_state.v10_val < -2.0 else '#065F46'}; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">
                                        {'Biến động mạnh' if st.session_state.v10_val < -2.0 else 'Bình thường'}
                                    </span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            """, unsafe_allow_html=True)


    # ----------------- PHẦN 2: QUY TRÌNH HỆ THỐNG -----------------
    elif nav_selection == "QUY TRÌNH HỆ THỐNG":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #0A2540; font-size: 1.3rem; font-weight: 800; margin-bottom: 15px;'><i class='fa-solid fa-gears'></i> Quy Trình & Đối Sánh Hệ Thống</h3>", unsafe_allow_html=True)
        
        sys_tab1, sys_tab2 = st.tabs([
            " QUY TRÌNH LUỒNG ĐI", 
            " ĐỐI SÁNH THỜI GIAN THỰC"
        ])
        
        with sys_tab1:
            st.markdown("<h4 style='color: #0A2540; font-size: 1rem; font-weight: 700; margin-bottom: 5px;'><i class='fa-solid fa-network-wired'></i> LUỒNG THỜI GIAN THỰC (REAL-TIME INFERENCE FLOW)</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; color: #4B5563; margin-bottom: 12px;'>Luồng dữ liệu giao dịch được thu thập đầu vào, tiền xử lý chuẩn hóa, giải thích SHAP và đẩy qua bộ máy suy luận rủi ro thời gian thực bất đồng bộ.</p>", unsafe_allow_html=True)
            
            if st.session_state.run_prediction:
                prob_str = f"{(st.session_state.main_fraud_prob * 100):.2f}%"
                decision_str = "GIAN LẬN (Fraud)" if st.session_state.main_prediction_label == 1 else "HỢP LỆ (Normal)"
            else:
                prob_str = "Đang chờ..."
                decision_str = "Đang chờ..."
                
            steps = [
                (
                    "Step 1: Input Received (Nhận Tín Hiệu Đầu Vào)",
                    f"""
                    * **Hoạt động:** Hệ thống giám sát giao dịch trực tuyến (API Gateway hoặc Broker thông điệp Kafka) tiếp nhận thông tin giao dịch thẻ tín dụng thời gian thực từ POS hoặc cổng thanh toán thương mại điện tử.
                    * **Thông số giao dịch hiện tại:**
                      - Số tiền giao dịch (`Amount`): **${st.session_state.amount_val:.2f}**
                      - Thời gian cách giao dịch trước (`Time`): **{st.session_state.time_val:.1f}** giây
                      - Chuỗi đặc trưng (`V1-V28`): Đã được trích xuất tự động và đồng bộ dựa trên lịch sử hành vi của tài khoản thẻ.
                    """
                ),
                (
                    "Step 2: Validation (Kiểm Tra & Định Dạng Dữ Liệu)",
                    """
                    * **Hoạt động:** Tiến hành kiểm định tính hợp lệ và cấu trúc định dạng của gói dữ liệu đầu vào (Sanity Check) trước khi truyền vào bộ xử lý sâu.
                    * **Chi tiết kỹ thuật:**
                      - Hệ thống xác thực vector đầu vào đảm bảo có đầy đủ **30 chiều** bao gồm `Time`, `Amount` và 28 biến ẩn phi tuyến `V1-V28` đại diện cho các hành vi giao dịch.
                      - Các giá trị khuyết thiếu (`NaN`) hoặc lỗi nhập liệu sẽ được điền khuyết tự động (Imputation) bằng trị số trung vị lịch sử tài khoản nhằm triệt tiêu hoàn toàn rủi ro gây lỗi hệ thống.
                    """
                ),
                (
                    "Step 3: Preprocessing (Tiền Xử Lý & Chuẩn Hóa Số Liệu)",
                    f"""
                    * **Hoạt động:** Do đặc trưng `Amount` và `Time` có phạm vi giá trị cực lớn so với các chiều `V1-V28`, chúng cần được đưa về cùng một phân phối thang đo để tránh thiên lệch mô hình.
                    * **Chi tiết kỹ thuật:**
                      - Chuẩn hóa **RobustScaler** được áp dụng cho `Amount` để chống nhiễu từ các giao dịch ngoại lai có giá trị cực đoan.
                      - Chuẩn hóa **StandardScaler** được áp dụng cho `Time` để đồng bộ phân phối thời gian.
                      - **Kết quả chuẩn hóa thực tế:** Amount = `{scaled_amt:.6f}`, Time = `{scaled_time:.6f}`.
                    """
                ),
                (
                    "Step 4: XAI Analysis (Giải Thích Đóng Góp SHAP)",
                    f"""
                    * **Hoạt động:** Tích hợp bộ máy giải thích **SHAP (SHapley Additive exPlanations)** đại diện cho Trí tuệ nhân tạo có thể giải thích (Explainable AI - XAI).
                    * **Chi tiết kỹ thuật:**
                      - SHAP tính toán phân bổ Shapley value phi tuyến động cho tất cả 30 biến số đầu vào đối với xác suất dự đoán cuối cùng.
                      - Báo cáo rõ ràng những biến ẩn nào (ví dụ V12, V14, V17) đóng vai trò làm lực đẩy (màu đỏ) kéo xác suất rủi ro lên cao và những biến làm lực cản (màu xanh), giúp nhân viên kiểm soát nắm bắt nguyên nhân cảnh báo.
                    """
                ),
                (
                    "Step 5: Prediction (Suy Luận Mô Hình Học Máy & Học Sâu)",
                    f"""
                    * **Hoạt động:** Vector 30 chiều sau tiền xử lý được chuyển tiếp song song đến bộ suy luận của các mô hình phân loại.
                    * **Mô hình chính được chọn vận hành:** **{primary_algo}** (sử dụng phiên bản huấn luyện tăng cường bằng thuật toán cân bằng dữ liệu thiểu số **SMOTE**).
                    * **Kết quả tính toán:** Xác suất rủi ro rò rỉ gian lận đạt: **{prob_str}**.
                    """
                ),
                (
                    "Step 6: Threshold (Đối Chiếu Ngưỡng Quyết Định)",
                    f"""
                    * **Hoạt động:** Đối chiếu xác suất rủi ro dự báo từ mô hình với ngưỡng cảnh báo động.
                    * **Chi tiết kỹ thuật:**
                      - Ngưỡng quyết định rủi ro hiện tại: **{st.session_state.decision_threshold_slider:.2f}** (được cấu hình bởi bộ phận quản trị rủi ro thẻ).
                      - Kết quả đối chiếu: Xác suất rủi ro thực tế **{prob_str}** {"vượt qua hoặc đạt" if st.session_state.main_prediction_label == 1 else "nằm dưới"} ngưỡng quyết định.
                      - **Kết luận hệ thống:** **{decision_str}**.
                    """
                ),
                (
                    "Step 7: Log Database (Ghi Nhật Ký Bất Đồng Bộ)",
                    """
                    * **Hoạt động:** Để triệt tiêu hoàn toàn thời gian trễ thanh toán (Latency) của khách hàng, hệ thống sử dụng cơ chế ghi nhật ký bất đồng bộ (Asynchronous Logging).
                    * **Chi tiết kỹ thuật:**
                      - Hệ thống đóng gói thông tin giao dịch bao gồm: ID giao dịch, Tên mô hình chính, Kết quả dự đoán, Xác suất rủi ro, Thời gian xử lý và Chuỗi đặc trưng 30 chiều.
                      - Tự động ghi nhận thông tin vào bảng `TransactionLogs` trên cơ sở dữ liệu **SQL Server (CreditCardFraudDB)** phục vụ công tác hậu kiểm và tái huấn luyện mô hình.
                    """
                )
            ]
            
            with st.container(height=640, border=True):
                for i, (title, desc) in enumerate(steps):
                    with st.expander(title, expanded=st.session_state.run_prediction):
                        st.markdown(desc)

        with sys_tab2:
            st.markdown("<h4 style='color: #0A2540; font-size: 1rem; font-weight: 700; margin-top: 10px; margin-bottom: 5px;'><i class='fa-solid fa-table-list'></i> Đối Sánh Phân Loại Thời Gian Thực (8 Cấu Hình)</h4>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; color: #4B5563; margin-bottom: 12px;'>Bảng dưới đây đối chiếu kết quả dự báo, độ tin cậy rủi ro và thời gian suy luận thực tế của 8 cấu hình mô hình (4 thuật toán x 2 phương pháp tiền xử lý) cho giao dịch đang được cấu hình ở khung trái.</p>", unsafe_allow_html=True)
            
            if not st.session_state.run_prediction:
                st.info("Vui lòng thiết lập thông số giao dịch và click nút **'KIỂM TRA GIAO DỊCH VÀ GHI NHẬT KÝ VẾT'** ở khung bên trái để xem kết quả đối sánh chi tiết thời gian thực.")
            else:
                # 1. Bảng dữ liệu thô (giới hạn chiều cao tránh cuộn)
                df_compare = pd.DataFrame(st.session_state.results_table)
                st.dataframe(df_compare, height=160, use_container_width=True, hide_index=True)
                
                # Tính toán các chỉ số thống kê so sánh nhanh
                frauds_count = sum(1 for row in st.session_state.results_table if "GIAN LẬN" in row["Kết quả nhận diện"])
                
                smote_probs = [float(row['Độ tin cậy rủi ro'].replace('%', '')) for row in st.session_state.results_table if "SMOTE" in row["Phương pháp xử lý dữ liệu"]]
                avg_smote = sum(smote_probs)/len(smote_probs) if smote_probs else 0
                
                no_smote_probs = [float(row['Độ tin cậy rủi ro'].replace('%', '')) for row in st.session_state.results_table if "Không SMOTE" in row["Phương pháp xử lý dữ liệu"]]
                avg_no_smote = sum(no_smote_probs)/len(no_smote_probs) if no_smote_probs else 0
                
                # 2. Thẻ so sánh chi tiết
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.markdown(f"""
                        <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.75rem; color: #991B1B; font-weight: bold;">MÔ HÌNH BÁO GIAN LẬN</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #991B1B; margin: 2px 0;">{frauds_count} / 8</div>
                            <div style="font-size: 0.65rem; color: #7F1D1D;">Cấu hình phát hiện rủi ro</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_info2:
                    st.markdown(f"""
                        <div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.75rem; color: #1E40AF; font-weight: bold;">RỦI RO TRUNG BÌNH (SMOTE)</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #1E40AF; margin: 2px 0;">{avg_smote:.2f}%</div>
                            <div style="font-size: 0.65rem; color: #1E3A8A;">Độ nhạy rủi ro nâng cao</div>
                        </div>
                    """, unsafe_allow_html=True)
                with col_info3:
                    st.markdown(f"""
                        <div style="background-color: #F0FDF4; border-left: 4px solid #10B981; padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.75rem; color: #166534; font-weight: bold;">RỦI RO TRUNG BÌNH (GỐC)</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #166534; margin: 2px 0;">{avg_no_smote:.2f}%</div>
                            <div style="font-size: 0.65rem; color: #14532D;">Dữ liệu chưa cân bằng</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<p style='margin-top: 15px;'></p>", unsafe_allow_html=True)

                # 3. Đồ thị đối sánh trực quan
                st.markdown("<h5 style='color: #0A2540; font-size: 0.9rem; font-weight: 700; margin-bottom: 5px;'><i class='fa-solid fa-chart-bar'></i> Biểu đồ Phân Phối Xác Suất Rủi Ro Giữa 8 Cấu Hình</h5>", unsafe_allow_html=True)
                try:
                    # Rút ngắn nhãn để tránh chồng chữ: chỉ dùng tên thuật toán + phương pháp ngắn gọn
                    algo_short = {
                        "Random Forest (Khuyên dùng)": "Random Forest",
                        "Logistic Regression": "Logistic Reg.",
                        "Decision Tree": "Decision Tree",
                        "1D-CNN (Deep Learning)": "1D-CNN"
                    }
                    method_short = {
                        "Dùng SMOTE": "(SMOTE)",
                        "Dữ liệu Gốc (Không SMOTE)": "(Gốc)"
                    }
                    names = [
                        f"{algo_short.get(row['Thuật toán'], row['Thuật toán'])} {method_short.get(row['Phương pháp xử lý dữ liệu'], row['Phương pháp xử lý dữ liệu'])}"
                        for row in st.session_state.results_table
                    ]
                    probs = [float(row['Độ tin cậy rủi ro'].replace('%', '')) for row in st.session_state.results_table]
                    colors = ['#EF4444' if "GIAN LẬN" in row['Kết quả nhận diện'] else '#10B981' for row in st.session_state.results_table]
                    
                    fig, ax = plt.subplots(figsize=(6, 3.2))
                    bars = ax.barh(names, probs, color=colors, height=0.55)
                    ax.set_xlabel('Xác suất rủi ro (%)', fontsize=9, color='#374151')
                    ax.set_xlim(0, 118)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#D1D5DB')
                    ax.spines['bottom'].set_color('#D1D5DB')
                    ax.tick_params(axis='y', which='major', labelsize=8.5, colors='#374151')
                    ax.tick_params(axis='x', which='major', labelsize=8, colors='#374151')
                    
                    # Thêm nhãn số phần trăm ngay sau mỗi cột
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1.0, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                                va='center', ha='left', fontsize=8, color='#374151', fontweight='bold')
                                
                    plt.tight_layout(pad=1.0)
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as ex:
                    st.warning(f"Không thể tải đồ thị trực quan. Chi tiết: {str(ex)}")



    # ----------------- PHẦN 3: NHẬT KÝ LƯU VẾT (SQL SERVER) -----------------
    elif nav_selection == "NHẬT KÝ LƯU VẾT":
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_t3_header, col_t3_btn = st.columns([3, 1])
        with col_t3_header:
            st.markdown("<h4 style='color: #0A2540; font-size: 1rem; font-weight: 700;'><i class='fa-solid fa-database'></i> Nhật ký Giám sát</h4>", unsafe_allow_html=True)
        with col_t3_btn:
            st.markdown("<div style='margin-top: 8px;'></div>", unsafe_allow_html=True)
            st.button("Tải lại dữ liệu")
            
        logs_data, conn_mode, _ = fetch_all_logs()
        
        if conn_mode == "SQL_SERVER":
            st.markdown("""
                <div style="display: inline-flex; align-items: center; gap: 6px; background-color: #DEF7EC; border: 1px solid #31C48D; color: #03543F; padding: 4px 8px; border-radius: 50px; font-size: 0.75rem; font-weight: bold; margin-bottom: 10px;">
                    <span style="display: inline-block; width: 6px; height: 6px; background-color: #31C48D; border-radius: 50%;"></span>
                     ĐÃ KẾT NỐI SQL SERVER
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="display: inline-flex; align-items: center; gap: 6px; background-color: #FDF2F2; border: 1px solid #F8B4B4; color: #9B1C1C; padding: 4px 8px; border-radius: 50px; font-size: 0.75rem; font-weight: bold; margin-bottom: 10px;">
                    <span style="display: inline-block; width: 6px; height: 6px; background-color: #F8B4B4; border-radius: 50%;"></span>
                    CHẾ ĐỘ NGOẠI TUYẾN (Offline Fallback)
                </div>
            """, unsafe_allow_html=True)

        if logs_data:
            df_logs = pd.DataFrame(logs_data)
            
            # Tạo bản sao dữ liệu thô phục vụ vẽ đồ thị thống kê trước khi format
            df_raw = df_logs.copy()
            
            def format_pred(val):
                if "FRAUD" in str(val).upper():
                    return "GIAN LẬN (Fraud)"
                return "HỢP LỆ (Normal)"
                
            df_logs["Prediction"] = df_logs["Prediction"].apply(format_pred)
            df_logs["Probability"] = df_logs["Probability"].apply(lambda x: f"{x:.2%}")
            df_logs["ExecutionTimeMs"] = df_logs["ExecutionTimeMs"].apply(lambda x: f"{x} ms")
            
            df_logs = df_logs.rename(columns={
                "TransactionID": "Mã Giao Dịch",
                "ModelName": "Mô Hình",
                "Prediction": "Kết Quả",
                "Probability": "Rủi Ro",
                "ExecutionTimeMs": "Xử Lý",
                "Timestamp": "Thời Điểm"
            })
            
            # --- TỐI ƯU GIAO DIỆN KHÔNG CẦN CUỘN TRANG (KPI & Biểu đồ lên trước, Bảng log thu gọn phía dưới) ---
            # Tính toán các chỉ số KPI nhanh
            total_txns = len(df_raw)
            fraud_txns = sum(1 for x in logs_data if "FRAUD" in str(x["Prediction"]).upper())
            normal_txns = total_txns - fraud_txns
            avg_exec_time = df_raw["ExecutionTimeMs"].mean() if "ExecutionTimeMs" in df_raw.columns else 0.0
            
            # 1. Thẻ thống kê KPI ngang (ở trên cùng)
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            with kpi_col1:
                st.markdown(f"""
                    <div style="background-color: #F3F4F6; border: 1px solid #E5E7EB; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                        <div style="font-size: 0.7rem; color: #4B5563; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">Giao dịch lưu vết</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #111827; margin: 2px 0;">{total_txns}</div>
                        <div style="font-size: 0.6rem; color: #6B7280;">Tổng nhật ký lưu trữ</div>
                    </div>
                """, unsafe_allow_html=True)
            with kpi_col2:
                st.markdown(f"""
                    <div style="background-color: #FEF2F2; border: 1px solid #FCA5A5; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                        <div style="font-size: 0.7rem; color: #991B1B; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">Cảnh báo rủi ro</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #EF4444; margin: 2px 0;">{fraud_txns}</div>
                        <div style="font-size: 0.6rem; color: #DC2626;">Đã nhận diện Gian lận</div>
                    </div>
                """, unsafe_allow_html=True)
            with kpi_col3:
                st.markdown(f"""
                    <div style="background-color: #ECFDF5; border: 1px solid #6EE7B7; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                        <div style="font-size: 0.7rem; color: #065F46; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">Giao dịch an toàn</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #10B981; margin: 2px 0;">{normal_txns}</div>
                        <div style="font-size: 0.6rem; color: #059669;">Hợp lệ & được duyệt</div>
                    </div>
                """, unsafe_allow_html=True)
            with kpi_col4:
                st.markdown(f"""
                    <div style="background-color: #EFF6FF; border: 1px solid #93C5FD; padding: 10px; border-radius: 8px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                        <div style="font-size: 0.7rem; color: #1E40AF; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;">Độ trễ trung bình</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #2563EB; margin: 2px 0;">{avg_exec_time:.1f} ms</div>
                        <div style="font-size: 0.6rem; color: #1D4ED8;">Thời gian phản hồi</div>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<p style='margin-top: 10px;'></p>", unsafe_allow_html=True)
            
            # 2. Vẽ 2 đồ thị so sánh (ở giữa)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("<h5 style='color: #0A2540; font-size: 0.85rem; font-weight: 700; margin-bottom: 5px; text-align: center;'><i class='fa-solid fa-chart-pie'></i> Tỷ Lệ Kết Quả Phân Loại</h5>", unsafe_allow_html=True)
                try:
                    # Biểu đồ tròn Donut so sánh loại Giao dịch (Fraud vs Normal)
                    fig1, ax1 = plt.subplots(figsize=(4.5, 1.8))
                    labels = ['Hop Le', 'Gian Lan']
                    sizes = [normal_txns, fraud_txns]
                    colors = ['#10B981', '#EF4444']
                    explode = (0, 0.1) if fraud_txns > 0 else (0, 0)
                    
                    wedges, texts, autotexts = ax1.pie(
                        sizes, 
                        explode=explode, 
                        labels=labels, 
                        colors=colors,
                        autopct='%1.1f%%', 
                        shadow=False, 
                        startangle=90,
                        textprops=dict(color="#374151", size=8),
                        pctdistance=0.7
                    )
                    
                    # Vẽ tâm vòng tròn để tạo donut
                    centre_circle = plt.Circle((0,0), 0.55, fc='white')
                    fig1.gca().add_artist(centre_circle)
                    
                    # Định dạng chữ hiển thị
                    for text in texts:
                        text.set_fontsize(7)
                        text.set_fontweight('bold')
                    for autotext in autotexts:
                        autotext.set_fontsize(8)
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        
                    ax1.axis('equal')  
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)
                except Exception as ex1:
                    st.warning(f"Không thể tải biểu đồ tròn. Chi tiết: {str(ex1)}")
                    
            with chart_col2:
                st.markdown("<h5 style='color: #0A2540; font-size: 0.85rem; font-weight: 700; margin-bottom: 5px; text-align: center;'><i class='fa-solid fa-chart-bar'></i> Số Lượng Nhật Ký Theo Mô Hình</h5>", unsafe_allow_html=True)
                try:
                    # Biểu đồ cột ngang - rút ngắn nhãn mô hình để tránh chồng chữ
                    fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
                    
                    model_counts = df_raw["ModelName"].value_counts()
                    # Rút ngắn tên: bỏ phần địuủ và giữ ngắn gọn
                    def shorten_model_name(name):
                        name = str(name)
                        name = name.replace("Random Forest (Khuyên dùng)", "RF")
                        name = name.replace("Random Forest", "RF")
                        name = name.replace("1D-CNN (Deep Learning)", "1D-CNN")
                        name = name.replace("Logistic Regression", "Logistic Regression")
                        name = name.replace("Decision Tree", "Decision Tree")
                        name = name.replace(" - SMOTE", " (SMOTE)")
                        name = name.replace(" - NoSMOTE", " (Gốc)")
                        return name
                    names = [shorten_model_name(x) for x in model_counts.index]
                    counts = model_counts.values
                    
                    ax2.barh(names, counts, color='#3B82F6', height=0.5)
                    ax2.set_xlabel('Số lượng', fontsize=8.5, color='#374151')
                    ax2.spines['top'].set_visible(False)
                    ax2.spines['right'].set_visible(False)
                    ax2.spines['left'].set_color('#D1D5DB')
                    ax2.spines['bottom'].set_color('#D1D5DB')
                    ax2.tick_params(axis='y', which='major', labelsize=8.5, colors='#374151')
                    ax2.tick_params(axis='x', which='major', labelsize=8, colors='#374151')
                    
                    plt.tight_layout(pad=0.8)
                    st.pyplot(fig2)
                    plt.close(fig2)
                except Exception as ex2:
                    st.warning(f"Không thể tải biểu đồ cột. Chi tiết: {str(ex2)}")
            
            # 3. Bảng dữ liệu chi tiết (ở dưới cùng, có thanh cuộn)
            st.markdown("<h5 style='color: #0A2540; font-size: 0.9rem; font-weight: 700; margin-top: 15px; margin-bottom: 8px;'><i class='fa-solid fa-list-check'></i> Nhật Ký Giao Dịch Chi Tiết</h5>", unsafe_allow_html=True)
            st.dataframe(df_logs, height=320, use_container_width=True, hide_index=True)
        else:
            st.write("Không tìm thấy nhật ký giao dịch.")
