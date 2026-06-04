import os
import sys
import logging
import html
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
    padding-top: 1.6rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
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
[data-testid="stSidebar"]{min-width:260px !important;max-width:260px !important;width:260px !important;overflow:hidden !important;display:none !important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]{overflow-y:hidden !important;overflow-x:hidden !important;}
[data-testid="stSidebar"]::-webkit-scrollbar, [data-testid="stSidebar"] *::-webkit-scrollbar{display:none !important;}
[data-testid="stSidebar"], [data-testid="stSidebar"] *{-ms-overflow-style:none !important;scrollbar-width:none !important;}
[data-testid="stSidebarDragHandle"], [data-testid="stSidebarResizer"], [class*="stSidebarDragHandle"], [class*="stSidebarResizer"]{display:none !important;pointer-events:none !important;}

/* Tối ưu hóa tabs ngang của Streamlit */
button[data-baseweb="tab"] {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: #4B5563 !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: #1E3A8A !important;
    background-color: rgba(30, 58, 138, 0.04) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1E3A8A !important;
    border-bottom: 3px solid #1E3A8A !important;
    background-color: rgba(30, 58, 138, 0.02) !important;
}

/* Dịch toàn bộ tabs bên phải lên cao để cân đối với cột trái */
div[data-testid="stTabs"] {
    margin-top: -30px !important;
}

/* Keep long transaction details compact and scrollable inside the dialog. */
div[data-testid="stDialog"] div[role="dialog"] {
    max-height: 68vh !important;
    overflow-y: auto !important;
}

.inference-decision-grid {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    align-items: center;
    gap: 12px;
    margin: 14px 0 10px;
}
.inference-metric-box {
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    background: #EFF6FF;
    padding: 10px;
    text-align: center;
}
.inference-metric-box.threshold {
    border-color: #DDD6FE;
    background: #F5F3FF;
}
.inference-metric-label {
    color: #64748B;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.inference-metric-value {
    color: #1E3A8A;
    font-size: 1.35rem;
    font-weight: 800;
    margin-top: 2px;
}
.inference-operator {
    color: #475569;
    font-size: 1.5rem;
    font-weight: 800;
}
.inference-risk-track {
    position: relative;
    height: 14px;
    overflow: visible;
    border-radius: 999px;
    background: #E5E7EB;
    margin: 12px 0 24px;
}
.inference-risk-fill {
    height: 100%;
    border-radius: 999px;
}
.inference-final {
    border-radius: 10px;
    color: #FFFFFF;
    padding: 11px 14px;
    text-align: center;
    font-size: 0.85rem;
    font-weight: 800;
}
.inference-detail {
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    background: #FFFFFF;
    margin-bottom: 8px;
    overflow: hidden;
}
.inference-detail summary {
    cursor: pointer;
    list-style: none;
    padding: 10px 12px;
    color: #0F172A;
    background: #F8FAFC;
    font-size: 0.8rem;
    font-weight: 800;
}
.inference-detail summary::-webkit-details-marker { display: none; }
.inference-detail-body {
    padding: 11px 12px;
    color: #475569;
    font-size: 0.76rem;
    line-height: 1.5;
}
.inference-mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 8px;
}
.inference-mini-card {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    background: #F8FAFC;
    padding: 8px;
}
.inference-code {
    overflow-x: auto;
    border-radius: 8px;
    background: #0F172A;
    color: #A7F3D0;
    padding: 10px;
    font-family: monospace;
    font-size: 0.68rem;
    line-height: 1.45;
    white-space: pre-wrap;
}
.inference-empty {
    border: 1px dashed #93C5FD;
    border-radius: 12px;
    background: #EFF6FF;
    color: #1E40AF;
    padding: 20px;
    text-align: center;
    font-size: 0.84rem;
    line-height: 1.6;
}
.pipeline-reference-head {
    border-bottom: 1px solid #E5E7EB;
    margin-bottom: 10px;
    padding-bottom: 8px;
}
.pipeline-reference-title {
    color: #0F172A;
    font-size: 1rem;
    font-weight: 800;
}
.pipeline-reference-desc {
    color: #64748B;
    font-size: 0.76rem;
    line-height: 1.5;
    margin-top: 3px;
}
.pipeline-status {
    border: 1px solid #A7F3D0;
    border-radius: 8px;
    background: #ECFDF5;
    color: #047857;
    font-size: 0.75rem;
    line-height: 1.45;
    margin-bottom: 8px;
    padding: 8px 10px;
}
.pipeline-status.info {
    border-color: #BFDBFE;
    background: #EFF6FF;
    color: #1D4ED8;
}
.pipeline-insight {
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    background: #EFF6FF;
    color: #1E40AF;
    font-size: 0.73rem;
    line-height: 1.5;
    margin-bottom: 8px;
    padding: 8px 10px;
}
.pipeline-probability {
    color: #0F172A;
    font-size: 0.76rem;
    font-weight: 800;
    margin: 8px 0 4px;
}
.pipeline-probability-value {
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 4px;
    text-align: center;
}
.pipeline-formula {
    border: 1px dashed #93C5FD;
    border-radius: 8px;
    background: #F8FAFC;
    color: #334155;
    font-size: 0.72rem;
    line-height: 1.55;
    margin-top: 8px;
    padding: 8px 10px;
}
.pipeline-prep-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 12px;
    margin: 10px 0 16px;
}
.pipeline-prep-card {
    flex: 1 1 220px;
    min-width: 220px;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    background: #FFFFFF;
    padding: 16px 18px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
}
.pipeline-prep-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 5px 10px;
    border-radius: 999px;
    background: #E0F2FE;
    color: #0369A1;
    font-size: 0.72rem;
    font-weight: 800;
    text-transform: uppercase;
}
.pipeline-prep-title {
    color: #0F172A;
    font-size: 0.9rem;
    font-weight: 800;
    margin-top: 10px;
    line-height: 1.3;
}
.pipeline-prep-note {
    color: #475569;
    font-size: 0.78rem;
    line-height: 1.55;
    margin-top: 10px;
}
.pipeline-prep-arrow {
    flex: 0 0 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #2563EB;
    font-size: 1.1rem;
}
.pipeline-output {
    border: 1px solid #A7F3D0;
    border-radius: 8px;
    background: #ECFDF5;
    color: #047857;
    font-size: 0.73rem;
    line-height: 1.5;
    margin-top: 8px;
    padding: 8px 10px;
}
.shap-caption {
    color: #64748B;
    font-size: 0.75rem;
    line-height: 1.45;
    margin: 5px 0 7px;
}
.shap-ranking {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    background: #FFFFFF;
    margin: 7px 0 10px;
    overflow: hidden;
}
.shap-ranking-header {
    display: grid;
    grid-template-columns: 1.1fr 2.2fr 1.65fr;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-bottom: 1px solid #E2E8F0;
    background: #F8FAFC;
    color: #64748B;
    font-size: 0.68rem;
    font-weight: 700;
}
.shap-ranking-scale {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.shap-ranking-scale span:first-child {
    color: #059669;
    text-align: right;
}
.shap-ranking-scale span:last-child {
    color: #DC2626;
}
.shap-ranking-row {
    display: grid;
    grid-template-columns: 1.1fr 2.2fr 1.65fr;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid #F1F5F9;
    padding: 8px 10px;
}
.shap-ranking-row:last-child {
    border-bottom: none;
}
.shap-ranking-feature {
    color: #0F172A;
    font-size: 0.74rem;
    font-weight: 800;
}
.shap-ranking-value {
    color: #64748B;
    font-size: 0.68rem;
    font-weight: 500;
    margin-top: 2px;
}
.shap-ranking-track {
    display: grid;
    grid-template-columns: 1fr 1px 1fr;
    align-items: center;
    height: 20px;
    background: linear-gradient(90deg, #ECFDF5 0%, #F8FAFC 48%, #F8FAFC 52%, #FEF2F2 100%);
    border-radius: 999px;
}
.shap-ranking-axis {
    height: 26px;
    background: #475569;
}
.shap-ranking-side {
    display: flex;
    align-items: center;
    height: 100%;
}
.shap-ranking-side.left {
    justify-content: flex-end;
}
.shap-ranking-bar {
    height: 16px;
    border-radius: 999px;
}
.shap-ranking-effect {
    color: #475569;
    font-size: 0.69rem;
    line-height: 1.35;
}
.shap-ranking-effect strong {
    font-size: 0.72rem;
}
.st-key-log_filter_start div[data-baseweb="input"],
.st-key-log_filter_end div[data-baseweb="input"] {
    min-height: 44px !important;
    border: none !important;
    border-radius: 8px !important;
    background: #F8FAFC !important;
    box-shadow: none !important;
}
.st-key-log_filter_start div[data-baseweb="input"] > div,
.st-key-log_filter_end div[data-baseweb="input"] > div {
    background: transparent !important;
}
.st-key-log_filter_start input,
.st-key-log_filter_end input {
    min-width: 0 !important;
    padding-left: 12px !important;
    color: #0F172A !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
.st-key-log_filter_start button,
.st-key-log_filter_end button {
    display: none !important;
}
.st-key-log_filter_apply button {
    min-height: 44px !important;
    min-width: 68px !important;
    padding: 7px 12px !important;
    border-radius: 8px !important;
    background: #173B76 !important;
    font-size: 0.76rem !important;
    line-height: 1 !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
}
.st-key-log_filter_apply button p {
    white-space: nowrap !important;
}
.st-key-log_refresh button {
    min-height: 44px !important;
    min-width: 76px !important;
    padding: 7px 11px !important;
    border-radius: 8px !important;
    font-size: 0.72rem !important;
    line-height: 1 !important;
    text-transform: none !important;
    white-space: nowrap !important;
}
.st-key-log_refresh button p {
    white-space: nowrap !important;
}
/* Custom Clickable HTML Log Table */
.custom-log-table-header {
    display: grid;
    grid-template-columns: 1.6fr 3.1fr 1.7fr 0.8fr 0.8fr 2fr;
    background-color: #F8FAFC;
    border: 1px solid #E5E7EB;
    border-radius: 8px 8px 0 0;
    padding: 10px 15px;
    font-weight: 700;
    color: #4B5563;
    font-size: 0.82rem;
}
.custom-log-row {
    display: grid;
    grid-template-columns: 1.6fr 3.1fr 1.7fr 0.8fr 0.8fr 2fr;
    padding: 10px 15px;
    border-bottom: 1px solid #F3F4F6;
    text-decoration: none !important;
    color: #4B5563 !important;
    align-items: center;
    transition: background-color 0.15s ease, color 0.15s ease;
}
.custom-log-row:hover {
    background-color: #F1F5F9;
    color: #0F172A !important;
}
.custom-log-row:last-child {
    border-bottom: none;
}
.custom-log-cell-id {
    font-weight: 700;
    color: #111827;
}
.custom-log-cell-model {
    color: #64748B;
}
.custom-log-cell-result {
    font-weight: 600;
    color: #374151;
}
.custom-log-cell-risk {
    color: #374151;
}
.custom-log-cell-latency {
    color: #374151;
}
.custom-log-cell-time {
    color: #374151;
}
.st-key-logs_clickable_table {
    border: 1px solid #E5E7EB;
    border-top: none;
    border-radius: 0 0 8px 8px;
    background-color: #FFFFFF;
}
.st-key-logs_clickable_table div[class*="st-key-log_row_"] {
    position: relative !important;
}
.st-key-logs_clickable_table div[class*="st-key-log_row_"] div[data-testid="stElementContainer"]:has(button),
.st-key-logs_clickable_table div[class*="st-key-log_row_"] div.element-container:has(div.stButton) {
    position: absolute !important;
    inset: 0 !important;
    z-index: 99999 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    height: 100% !important;
}
.st-key-logs_clickable_table div[class*="st-key-log_row_"] button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    border: none !important;
    background: transparent !important;
    opacity: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    cursor: pointer !important;
    z-index: 99999 !important;
}
.st-key-logs_clickable_table div[class*="st-key-log_row_"]:hover .custom-log-row {
    background-color: #F1F5F9;
    color: #0F172A !important;
}
@media (max-width: 1100px) {
    .pipeline-prep-flow {
        display: block;
    }
    .pipeline-prep-card {
        margin-bottom: 7px;
    }
    .pipeline-prep-arrow {
        display: none;
    }
    .shap-ranking-row {
        grid-template-columns: 1fr;
        gap: 5px;
    }
    .shap-ranking-header {
        display: block;
    }
    .shap-ranking-header > span {
        display: none;
    }
}
</style>""", unsafe_allow_html=True)

# Thư mục chứa mô hình
MODEL_DIR = "models"
MODEL_DIR_FALLBACK = r"D:\Intern-Projects\credit-fraud-detection\models"
MODEL_FEATURE_NAMES = ["scaled_amount", "scaled_time"] + [f"V{i}" for i in range(1, 29)]

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
    # --- TỐI ƯU HÓA HIỆU NĂNG: Tránh kết nối lại liên tục nếu SQL Server đang offline ---
    now = time.time()
    if "sql_last_fail_time" in st.session_state:
        if now - st.session_state.sql_last_fail_time < 30.0:  # Cooldown 30 giây
            return None, "SQL Server đang tạm ngắt kết nối (chế độ offline). Đang bỏ qua kết nối để tối ưu tốc độ."

    drivers = [d for d in pyodbc.drivers() if 'SQL Server' in d or 'ODBC' in d]
    if not drivers:
        logger.warning("Không tìm thấy ODBC Driver SQL Server nào trên máy!")
        st.session_state.sql_last_fail_time = now
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
        # Giảm timeout xuống 1 giây để phản hồi ngay lập tức nếu server không khả dụng
        conn = pyodbc.connect(conn_str, timeout=1)
        if "sql_last_fail_time" in st.session_state:
            del st.session_state.sql_last_fail_time
        return conn, None
    except Exception as e:
        logger.error(f"Lỗi kết nối SQL Server: {str(e)}")
        st.session_state.sql_last_fail_time = now
        return None, f"Không thể kết nối đến SQL Server DANH-PC. Đang chuyển sang offline fallback. (Chi tiết: {str(e)})"

def insert_log_to_sql(transaction_id, model_name, prediction, probability, execution_time_ms, input_time, input_amount, v_values):
    """Ghi dữ liệu kiểm thử vào SQL Server, tự động chuyển về local state nếu ngoại tuyến."""
    v_values_str = ",".join([f"{v:.4f}" for v in v_values])

    conn, _ = get_sql_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()

            # 1. Ghi vào bảng chính TransactionLogs
            query = """
                INSERT INTO TransactionLogs (TransactionID, ModelName, Prediction, Probability, ExecutionTimeMs, InputTime, InputAmount, V_Values)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(query, (transaction_id, model_name, prediction, probability, execution_time_ms, input_time, input_amount, v_values_str))

            # 2. Ghi vào bảng chi tiết TransactionLog_Detail
            query_detail = """
                INSERT INTO TransactionLog_Detail (TransactionID, Time, Amount,
                    V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,
                    V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28)
                VALUES (?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            try:
                digits_only = "".join(filter(str.isdigit, transaction_id))
                txn_id_int = int(digits_only) if digits_only else 0
            except Exception:
                txn_id_int = 0

            time_str = f"{input_time:.1f}"
            amount_str = f"{input_amount:.2f}"

            params_detail = [txn_id_int, time_str, amount_str] + list(v_values)
            cursor.execute(query_detail, params_detail)

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
    except Exception:
        pass
    return full_logs, "OFFLINE_FALLBACK", err

def fetch_log_detail(transaction_id):
    """Truy vấn thông tin chi tiết một giao dịch từ SQL Server hoặc session state."""
    digits_only = "".join(filter(str.isdigit, str(transaction_id)))
    detail_transaction_id = int(digits_only) if digits_only else 0

    # 1. Tìm trong local session logs trước
    if "local_logs" in st.session_state:
        for log in st.session_state.local_logs:
            if log.get("TransactionID") == transaction_id:
                return {
                    "RowID": log.get("RowID", "-"),
                    "DetailTransactionID": log.get("DetailTransactionID", detail_transaction_id),
                    "TransactionID": log.get("TransactionID"),
                    "ModelName": log.get("ModelName"),
                    "Prediction": log.get("Prediction"),
                    "Probability": log.get("Probability"),
                    "ExecutionTimeMs": log.get("ExecutionTimeMs"),
                    "InputTime": log.get("InputTime", 406.0),
                    "InputAmount": log.get("InputAmount", 4.90),
                    "V_Values": log.get("V_Values", ",".join(["0.0"]*28)),
                    "Timestamp": log.get("Timestamp")
                }

    # 2. Truy vấn từ SQL Server
    conn, _ = get_sql_connection()
    if conn is not None:
        try:
            cursor = conn.cursor()
            # Lấy thông tin cơ bản từ TransactionLogs
            query_log = """
                SELECT TransactionID, ModelName, Prediction, Probability, ExecutionTimeMs, InputTime, InputAmount, V_Values, Timestamp
                FROM TransactionLogs
                WHERE TransactionID = ?
            """
            cursor.execute(query_log, (transaction_id,))
            r_log = cursor.fetchone()

            if r_log:
                # Khớp với TransactionID dạng int trong TransactionLog_Detail.
                txn_id_int = detail_transaction_id
                row_id = "-"
                detail_txn_id = txn_id_int
                time_val = None
                amount_val = None
                v_values_str = None

                # Cố gắng lấy chi tiết từ TransactionLog_Detail
                if txn_id_int > 0:
                    try:
                        query_detail = """
                            SELECT RowID, TransactionID, Time, Amount,
                                   V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,
                                   V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28
                            FROM TransactionLog_Detail
                            WHERE TransactionID = ?
                            ORDER BY RowID DESC
                        """
                        cursor.execute(query_detail, (txn_id_int,))
                        r_det = cursor.fetchone()
                        if r_det:
                            row_id = r_det[0]
                            detail_txn_id = r_det[1]
                            time_val = r_det[2]
                            amount_val = r_det[3]
                            v_values_list = [f"{val:.4f}" if val is not None else "0.0000" for val in r_det[4:32]]
                            v_values_str = ",".join(v_values_list)
                    except Exception as ed:
                        logger.error(f"Lỗi lấy chi tiết từ TransactionLog_Detail: {str(ed)}")

                # Nếu không tìm thấy trong TransactionLog_Detail hoặc lỗi, dùng fallback từ TransactionLogs
                if time_val is None:
                    time_val = r_log[5] if r_log[5] is not None else 406.0
                if amount_val is None:
                    amount_val = r_log[6] if r_log[6] is not None else 4.90
                if v_values_str is None:
                    v_values_str = r_log[7] if r_log[7] is not None else ",".join(["0.0"]*28)

                conn.close()
                return {
                    "RowID": row_id,
                    "DetailTransactionID": detail_txn_id,
                    "TransactionID": r_log[0],
                    "ModelName": r_log[1],
                    "Prediction": r_log[2],
                    "Probability": r_log[3],
                    "ExecutionTimeMs": r_log[4],
                    "InputTime": time_val,
                    "InputAmount": amount_val,
                    "V_Values": v_values_str,
                    "Timestamp": r_log[8].strftime("%Y-%m-%d %H:%M:%S") if hasattr(r_log[8], "strftime") else str(r_log[8])
                }
            conn.close()
        except Exception as e:
            logger.error(f"Lỗi lấy chi tiết log từ SQL Server: {str(e)}")

    # 3. Fallback tìm trong static logs
    static_logs = [
        {"TransactionID": "TXN-006", "ModelName": "Random Forest - NoSMOTE", "Prediction": "NORMAL", "Probability": 0.0421, "ExecutionTimeMs": 45, "Timestamp": "2026-05-26 19:08:15", "InputTime": 406.0, "InputAmount": 4.90, "V_Values": ",".join(["0.0"]*28)},
        {"TransactionID": "TXN-005", "ModelName": "1D-CNN - SMOTE", "Prediction": "NORMAL", "Probability": 0.0115, "ExecutionTimeMs": 68, "Timestamp": "2026-05-26 18:55:10", "InputTime": 500.0, "InputAmount": 120.0, "V_Values": ",".join(["0.0"]*28)}
    ]
    for log in static_logs:
        if log["TransactionID"] == transaction_id:
            return {
                **log,
                "RowID": "-",
                "DetailTransactionID": detail_transaction_id
            }

    return None

def render_detail_ui(txn_id):
    detail = fetch_log_detail(txn_id)
    if not detail:
        st.error(f"Không tìm thấy thông tin cho mã giao dịch {txn_id}.")
        return

    is_fraud = "FRAUD" in str(detail['Prediction']).upper()
    status_text = "CẢNH BÁO GIAN LẬN" if is_fraud else "GIAO DỊCH HỢP LỆ"
    status_color = "#EF4444" if is_fraud else "#10B981"
    status_bg = "#FEF2F2" if is_fraud else "#ECFDF5"
    status_border = "#FCA5A5" if is_fraud else "#A7F3D0"

    st.markdown(f"""
        <div style="font-size: 0.9rem; color: #6B7280; padding-bottom: 10px; border-bottom: 1px solid #E5E7EB; margin-bottom: 14px;">
            Mã tham chiếu UI: <strong style="color: #1E3A8A;">{detail['TransactionID']}</strong>
            <span style="color: #9CA3AF; padding: 0 6px;">|</span>
            Mô hình đánh giá: <strong style="color: #374151;">{detail['ModelName']}</strong>
        </div>
    """, unsafe_allow_html=True)

    # Metadata chính của giao dịch
    row_id = detail.get("RowID", "-")
    detail_transaction_id = detail.get("DetailTransactionID", "-")
    st.markdown("<div style='font-size: 0.88rem; color: #9CA3AF; font-weight: 800; margin-bottom: 8px;'>THÔNG TIN CHÍNH</div>", unsafe_allow_html=True)
    col_meta1, col_meta2, col_meta3, col_meta4 = st.columns(4)
    with col_meta1:
        st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #E5E7EB; padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.72rem; color: #4B5563; font-weight: bold;"><span style="color: #EAB308;">&#128273;</span> RowID</div>
                <div style="font-size: 1rem; font-weight: 700; color: #111827; margin-top: 4px;">{row_id}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_meta2:
        st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #E5E7EB; padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.72rem; color: #4B5563; font-weight: bold;">TransactionID</div>
                <div style="font-size: 1rem; font-weight: 700; color: #111827; margin-top: 4px;">{detail_transaction_id}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_meta3:
        st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #E5E7EB; padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.72rem; color: #4B5563; font-weight: bold;">Time</div>
                <div style="font-size: 1rem; font-weight: 700; color: #111827; margin-top: 4px;">{float(detail['InputTime']):.1f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_meta4:
        st.markdown(f"""
            <div style="background-color: #ffffff; border: 1px solid #93C5FD; padding: 10px; border-radius: 8px;">
                <div style="font-size: 0.72rem; color: #4B5563; font-weight: bold;">Amount</div>
                <div style="font-size: 1rem; font-weight: 800; color: #2563EB; margin-top: 4px;">{float(detail['InputAmount']):.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="font-size: 0.78rem; color: #6B7280; margin: 10px 0;">
            Thời điểm ghi nhận: <strong style="color: #374151;">{detail['Timestamp']}</strong>
            <span style="color: #D1D5DB; padding: 0 8px;">|</span>
            Độ trễ xử lý: <strong style="color: #2563EB;">{detail['ExecutionTimeMs']} ms</strong>
        </div>
        <div style="background-color: {status_bg}; border: 1px solid {status_border}; border-left: 5px solid {status_color}; padding: 12px 15px; border-radius: 8px; margin-bottom: 15px;">
            <div style="font-weight: 700; color: {status_color}; font-size: 1rem; margin-bottom: 3px;">
                {status_text} (Xác suất rủi ro: {float(detail['Probability']):.2%})
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Danh sách 28 biến đặc trưng V ẩn (V1-V28)
    try:
        v_vals = [float(x.strip()) for x in str(detail['V_Values']).split(",") if x.strip() != ""]
    except Exception:
        v_vals = [0.0] * 28

    if len(v_vals) == 28:
        with st.expander("🔍 Xem chi tiết 28 chỉ số biến ẩn (V1 - V28)", expanded=False):
            df_v = pd.DataFrame({
                "Biến số": [f"V{i}" for i in range(1, 29)],
                "Giá trị": v_vals
            })

            view_mode = st.selectbox(
                "Chế độ hiển thị:",
                ["Biểu đồ", "Bảng dữ liệu"],
                key=f"v_detail_view_{txn_id}"
            )

            if view_mode == "Biểu đồ":
                fig_v, ax_v = plt.subplots(figsize=(6.5, 2.2))
                colors_v = ['#EF4444' if x < -2.0 else ('#10B981' if x > 2.0 else '#6B7280') for x in v_vals]
                ax_v.bar(df_v["Biến số"], df_v["Giá trị"], color=colors_v, edgecolor='none', width=0.6)
                ax_v.axhline(0, color='#9CA3AF', linewidth=0.8, linestyle='--')
                ax_v.set_ylabel('Giá trị biến ẩn', fontsize=8.0, color='#4B5563')
                ax_v.spines['top'].set_visible(False)
                ax_v.spines['right'].set_visible(False)
                ax_v.spines['left'].set_color('#D1D5DB')
                ax_v.spines['bottom'].set_color('#D1D5DB')
                ax_v.tick_params(axis='y', which='major', labelsize=8.0, colors='#374151')
                ax_v.tick_params(axis='x', which='major', labelsize=7.0, colors='#374151')
                plt.xticks(rotation=0)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig_v)
                plt.close(fig_v)
            else:
                st.markdown("<h5 style='color: #0A2540; font-size: 0.85rem; font-weight: 700; margin-top: 10px; margin-bottom: 5px;'><i class='fa-solid fa-list'></i> Thông số chi tiết các biến V1 - V28</h5>", unsafe_allow_html=True)
                grid_html = "<table style='width: 100%; border-collapse: collapse; font-size: 0.76rem; text-align: left;'>"
                for r in range(7):  # 7 hàng
                    grid_html += "<tr style='border-bottom: 1px solid #F3F4F6;'>"
                    for c in range(4):  # 4 cột mỗi hàng
                        v_idx = r * 4 + c
                        v_name = f"V{v_idx + 1}"
                        v_val = v_vals[v_idx]
                        color = "#EF4444" if v_val < -2.0 else ("#10B981" if v_val > 2.0 else "#374151")
                        grid_html += f"<td style='padding: 3px 4px; font-weight: 600; color: #4B5563; width: 10%;'>{v_name}:</td>"
                        grid_html += f"<td style='padding: 3px 4px; font-weight: bold; color: {color}; width: 15%;'>{v_val:.4f}</td>"
                    grid_html += "</tr>"
                grid_html += "</table>"
                st.markdown(grid_html, unsafe_allow_html=True)

def reset_detail_selection():
    st.session_state.logs_detail_table_version = st.session_state.get("logs_detail_table_version", 0) + 1

def close_detail_modal():
    reset_detail_selection()
    st.rerun()

def render_clickable_logs_table(df_logs):
    """Hiển thị bảng nhật ký HTML; click một hàng để mở popup mà không điều hướng trang."""
    columns = ["Mã Giao Dịch", "Mô Hình", "Kết Quả", "Rủi Ro", "Xử Lý", "Thời Điểm"]
    header_html = "".join(f"<div>{html.escape(column)}</div>" for column in columns)
    st.markdown(f'<div class="custom-log-table-header">{header_html}</div>', unsafe_allow_html=True)

    table_version = st.session_state.get("logs_detail_table_version", 0)
    selected_id = None
    with st.container(height=320, border=False, key="logs_clickable_table"):
        for row_position, (_, row) in enumerate(df_logs.iterrows()):
            transaction_id = str(row["Mã Giao Dịch"])
            cells = [
                ("custom-log-cell-id", transaction_id),
                ("custom-log-cell-model", row["Mô Hình"]),
                ("custom-log-cell-result", row["Kết Quả"]),
                ("custom-log-cell-risk", row["Rủi Ro"]),
                ("custom-log-cell-latency", row["Xử Lý"]),
                ("custom-log-cell-time", row["Thời Điểm"]),
            ]
            cells_html = "".join(
                f'<div class="{css_class}">{html.escape(str(value))}</div>'
                for css_class, value in cells
            )
            with st.container(key=f"log_row_{row_position}"):
                st.markdown(f'<div class="custom-log-row">{cells_html}</div>', unsafe_allow_html=True)
                if st.button(
                    f"Xem chi tiết {transaction_id}",
                    key=f"log_detail_{table_version}_{row_position}_{transaction_id}"
                ):
                    selected_id = transaction_id

    return selected_id

# Hỗ trợ popup dialog theo phiên bản Streamlit
if hasattr(st, "dialog"):
    @st.dialog("Chi Tiết Nhật Ký Giao Dịch", width="large", on_dismiss=reset_detail_selection)
    def show_detail_modal(txn_id):
        render_detail_ui(txn_id)
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        if st.button("Đóng", type="primary", use_container_width=True):
            close_detail_modal()
else:
    def show_detail_modal(txn_id):
        st.markdown("<hr style='border: 1px solid #1E3A8A;'>", unsafe_allow_html=True)
        render_detail_ui(txn_id)
        if st.button("Đóng", type="secondary", use_container_width=True):
            close_detail_modal()

def render_shap_feature_explanations(top_features):
    """Gộp biểu đồ và giải thích SHAP thành một bảng xếp hạng ngắn gọn."""
    max_abs_shap = max(float(top_features["Absolute_SHAP"].max()), 1e-12)
    
    FEATURE_LABELS = {
        "scaled_amount": "Amount - Số tiền",
        "scaled_time": "Time - Thời điểm",
        "V14": "V14 - Độ bảo mật tài khoản",
        "V12": "V12 - Độ an toàn thẻ thanh toán",
        "V10": "V10 - Độ tin cậy giao dịch",
        "V4": "V4 - Tần suất giao dịch dồn dập",
        "V17": "V17 - Đặc điểm thiết bị/địa điểm",
        "V11": "V11 - Danh mục tiêu dùng rủi ro",
        "V3": "V3 - Lịch sử uy tín tài khoản",
        "V7": "V7 - Lưu lượng giao dịch bất thường",
        "V16": "V16 - Kênh thanh toán/Đối tác",
        "V2": "V2 - Số lượng yêu cầu/lần thử",
        "V18": "V18 - Hạn mức/Quy mô giao dịch",
        "V1": "V1 - Đặc trưng vị trí địa lý",
        "V9": "V9 - Hành vi truy cập hệ thống",
        "V25": "V25 - Tín hiệu mạng/IP giao dịch",
        "V22": "V22 - Phiên đăng nhập/Trình duyệt",
        "V8": "V8 - Trạng thái xác thực thẻ",
    }
    
    def get_friendly_explanation(feature_name, val, shap_val):
        is_risk_up = shap_val > 0
        if feature_name == "scaled_amount":
            amt = st.session_state.amount_val
            if amt > 500:
                return f"Số tiền giao dịch lớn (${amt:.2f}), vượt xa mức bình thường ($22.75). Điều này làm tăng sự bất thường và đẩy rủi ro gian lận lên."
            else:
                return f"Số tiền giao dịch nhỏ (${amt:.2f}), nằm gần mức bình thường ($22.75). Đây là tín hiệu an toàn giúp giảm rủi ro."
        elif feature_name == "scaled_time":
            time_val = st.session_state.time_val
            return f"Thời điểm giao dịch ({time_val:.1f} giây) được sử dụng để phân tích quy luật thời gian của hành vi."
        
        # Check hidden features
        v_num_str = feature_name.replace("V", "")
        if v_num_str.isdigit():
            v_idx = int(v_num_str)
            baselines = {14: -0.50, 12: -3.30, 10: -0.10, 4: 1.15}
            if v_idx in baselines:
                base = baselines[v_idx]
                if v_idx == 4:
                    if val > 3.0:
                        return f"Chỉ số tần suất V4 là {val:.2f} (tăng cao so với mốc {base:.2f}), báo hiệu tần suất giao dịch dồn dập đáng nghi, đẩy rủi ro lên."
                    else:
                        return f"Chỉ số tần suất V4 là {val:.2f} (gần mốc bình thường {base:.2f}), cho thấy tần suất bình thường, giúp giảm rủi ro."
                else: # V14, V12, V10
                    if val < (base - 1.5):
                        return f"Chỉ số bảo mật V{v_idx} tụt sâu xuống {val:.2f} (thấp hơn nhiều so với mốc an toàn {base:.2f}), nghi ngờ rò rỉ thông tin thẻ hoặc thiết bị lạ, đẩy rủi ro lên."
                    else:
                        return f"Chỉ số bảo mật V{v_idx} là {val:.2f} (gần mốc an toàn {base:.2f}), phản ánh trạng thái bình thường, kéo rủi ro xuống."
            
            # General V features
            if is_risk_up:
                return f"Chỉ số V{v_idx} có giá trị {val:.2f} lệch khỏi trạng thái ổn định thông thường, được AI nhận định là dấu hiệu bất thường làm tăng rủi ro."
            else:
                return f"Chỉ số V{v_idx} có giá trị {val:.2f} nằm trong tầm kiểm soát an toàn, góp phần kéo giảm rủi ro."
                
        return f"Chỉ số {feature_name} có giá trị {val:.2f}."

    rows = []
    for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
        feature = str(row["Feature"])
        value = float(row["Value"])
        shap_value = float(row["SHAP"])
        abs_shap = abs(shap_value)
        
        if shap_value > 0:
            direction = "Tăng rủi ro (đẩy rủi ro lên)"
            direction_color = "#DC2626"
            bar_color = "#EF4444"
        elif shap_value < 0:
            direction = "Giảm rủi ro (kéo về an toàn)"
            direction_color = "#059669"
            bar_color = "#10B981"
        else:
            direction = "Gần như trung tính"
            direction_color = "#64748B"
            bar_color = "#94A3B8"

        display_name = FEATURE_LABELS.get(feature, f"{feature} - Chỉ số ẩn danh")

        if feature == "scaled_amount":
            value_text = f"${st.session_state.amount_val:.2f} (chuẩn hóa: {value:.4f})"
        elif feature == "scaled_time":
            value_text = f"{st.session_state.time_val:.1f}s (chuẩn hóa: {value:.4f})"
        else:
            value_text = f"{value:.4f}"

        bar_width = abs_shap / max_abs_shap * 100
        min_width_style = "min-width: 48px;" if abs_shap > 1e-4 else ""
        if shap_value < 0:
            bar_html = f"""
    <div class="shap-ranking-side left">
        <div class="shap-ranking-bar" style="width: {bar_width:.2f}%; background: {bar_color}; {min_width_style} display: flex; align-items: center; justify-content: center; color: white; font-size: 0.63rem; font-weight: bold; line-height: 1; padding: 0 4px; box-sizing: border-box; text-shadow: 0 1px 1px rgba(0,0,0,0.15);">
            {shap_value:+.4f}
        </div>
    </div>
    <div class="shap-ranking-axis"></div>
    <div class="shap-ranking-side"></div>"""
        else:
            bar_html = f"""
    <div class="shap-ranking-side left"></div>
    <div class="shap-ranking-axis"></div>
    <div class="shap-ranking-side">
        <div class="shap-ranking-bar" style="width: {bar_width:.2f}%; background: {bar_color}; {min_width_style} display: flex; align-items: center; justify-content: center; color: white; font-size: 0.63rem; font-weight: bold; line-height: 1; padding: 0 4px; box-sizing: border-box; text-shadow: 0 1px 1px rgba(0,0,0,0.15);">
            {shap_value:+.4f}
        </div>
    </div>"""

        friendly_exp = get_friendly_explanation(feature, value, shap_value)
        rows.append(f"""
<div class="shap-ranking-row" style="display: block; border-bottom: 1px solid #F1F5F9; padding: 10px 12px; background: {'#FEF2F2' if shap_value > 0 else '#ECFDF5'}; margin-bottom: 6px; border-radius: 6px;">
    <div style="display: grid; grid-template-columns: 1.2fr 2fr 1.5fr; align-items: center; gap: 10px; margin-bottom: 6px;">
        <div>
            <div class="shap-ranking-feature" style="font-size: 0.78rem; font-weight: bold; color: #1E293B;">#{rank} {html.escape(display_name)}</div>
            <div class="shap-ranking-value" style="font-size: 0.7rem; color: #64748B; margin-top: 1px;">Giá trị thực: {html.escape(value_text)}</div>
        </div>
        <div class="shap-ranking-track">{bar_html}</div>
        <div class="shap-ranking-effect" style="text-align: right;">
            <span style="background-color: {'#FEE2E2' if shap_value > 0 else '#D1FAE5'}; color: {direction_color}; padding: 3px 6px; border-radius: 4px; font-weight: bold; font-size: 0.65rem;">
                {direction}
            </span>
        </div>
    </div>
    <div style="font-size: 0.74rem; color: #475569; padding: 6px 8px; background: #FFFFFF; border-radius: 4px; border-left: 3px solid {direction_color}; margin-top: 4px; line-height: 1.45; box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);">
        <strong>Lý giải của AI:</strong> {friendly_exp}
    </div>
</div>""")

    st.markdown("""
<div style="background-color: #F8FAFC; border-left: 4px solid #64748B; padding: 8px 12px; margin-bottom: 12px; border-radius: 4px; font-size: 0.75rem; color: #475569; line-height: 1.45;">
    <strong>💡 Hướng dẫn nhanh:</strong> Bảng dưới xếp hạng 5 biến số có tác động mạnh nhất đến quyết định của mô hình AI cho giao dịch này. 
    Các thanh màu đỏ đại diện cho các yếu tố đẩy rủi ro lên (hướng gian lận). Các thanh màu xanh đại diện cho các yếu tố kéo rủi ro xuống (hướng an toàn).
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="shap-ranking">
    <div class="shap-ranking-header">
        <span>Biến đang xét</span>
        <div class="shap-ranking-scale">
            <span>← Kéo giảm rủi ro</span>
            <span>Đẩy tăng rủi ro →</span>
        </div>
        <span>Kết quả tác động</span>
    </div>
    {"".join(rows)}
</div>
""", unsafe_allow_html=True)

def build_reference_comparison_html():
    """Tạo bảng đối chiếu minh họa, không tham gia vào kết quả dự đoán."""
    comparison_rows = [
        ("Amount - Số tiền", st.session_state.amount_val, 22.75, "$", "", 2, st.session_state.amount_val > 500, "Cao bất thường", "Bình thường"),
        ("Time - Thời gian", st.session_state.time_val, 406.0, "", "s", 1, False, "", "Bình thường"),
        ("V14 - Chỉ số ẩn", st.session_state.v14_val, -0.50, "", "", 2, st.session_state.v14_val < -3.0, "Giảm sâu", "Gần mốc tham khảo"),
        ("V12 - Chỉ số ẩn", st.session_state.v12_val, -3.30, "", "", 2, st.session_state.v12_val < -5.0, "Giảm sâu", "Gần mốc tham khảo"),
        ("V4 - Chỉ số ẩn", st.session_state.v4_val, 1.15, "", "", 2, st.session_state.v4_val > 3.0, "Tăng cao", "Gần mốc tham khảo"),
        ("V10 - Chỉ số ẩn", st.session_state.v10_val, -0.10, "", "", 2, st.session_state.v10_val < -2.0, "Giảm sâu", "Gần mốc tham khảo"),
    ]
    rows = []
    for label, current, reference, prefix, suffix, decimals, is_alert, alert_text, normal_text in comparison_rows:
        assessment = alert_text if is_alert else normal_text
        assessment_bg = "#FEE2E2" if is_alert else "#D1FAE5"
        assessment_color = "#991B1B" if is_alert else "#065F46"
        rows.append(f"""
<tr style="border-bottom: 1px solid #F3F4F6;">
    <td style="padding: 7px 8px; font-weight: 600;">{html.escape(label)}</td>
    <td style="padding: 7px 8px; font-weight: bold; color: #1F2937;">{prefix}{current:.{decimals}f}{suffix}</td>
    <td style="padding: 7px 8px; color: #6B7280;">{prefix}{reference:.{decimals}f}{suffix}</td>
    <td style="padding: 7px 8px; text-align: right;">
        <span style="background-color: {assessment_bg}; color: {assessment_color}; padding: 3px 7px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">
            {html.escape(assessment)}
        </span>
    </td>
</tr>""")

    return f"""
<div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 8px; color: #1E3A8A; font-size: 0.82rem; line-height: 1.6; margin-bottom: 12px; padding: 12px 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <h5 style="margin-top: 0; margin-bottom: 8px; color: #1D4ED8; font-size: 0.9rem; font-weight: 700; display: flex; align-items: center; gap: 6px;">
        <i class="fa-solid fa-circle-question"></i> Tại sao cần Mốc Tham Khảo (Baseline)?
    </h5>
    <p style="margin: 0 0 10px 0;">
        Giống như trong y tế, bác sĩ cần biết <strong>nhịp tim chuẩn lúc nghỉ ngơi</strong> (ví dụ: 60 - 100 nhịp/phút) để đánh giá nhịp tim hiện tại của bạn có bất thường hay không. 
        Trong hệ thống AI phát hiện gian lận của chúng tôi:
    </p>
    <div style="display: grid; grid-template-columns: 1fr; gap: 8px; margin-bottom: 12px; padding-left: 10px; border-left: 3px solid #60A5FA;">
        <div>
            <strong>1. Mốc tham khảo là gì?</strong> Đây là giá trị <strong>trung vị thống kê</strong> thu thập từ hàng chục ngàn giao dịch <strong>hợp lệ, an toàn</strong> trong lịch sử. Nó đại diện cho một giao dịch "bình thường".
        </div>
        <div>
            <strong>2. Mục đích đối chiếu:</strong> Khi một giao dịch mới xảy ra, AI sẽ đo lường mức độ <strong>lệch chuẩn</strong> của giao dịch này so với mốc an toàn. Khoảng cách lệch càng lớn, độ nghi ngờ rò rỉ tài khoản hay hành vi gian lận càng cao.
        </div>
    </div>
    <div style="background-color: #DBEAFE; border-radius: 6px; padding: 10px 12px; font-size: 0.78rem; color: #1E40AF; border: 1px dashed #93C5FD;">
        <strong>💡 Ví dụ thực tế từ bảng dưới:</strong>
        <ul style="margin: 4px 0 0 18px; padding: 0; list-style-type: disc;">
            <li><strong>Amount (Số tiền):</strong> Giao dịch thông thường chỉ khoảng <strong>$22.75</strong>. Nếu giao dịch hiện tại là <strong>$500.00</strong>, việc vượt mốc tham khảo này sẽ khiến AI nâng mức cảnh báo nguy cơ.</li>
            <li><strong>V14 (Chỉ số bảo mật ẩn):</strong> Giao dịch an toàn có chỉ số V14 quanh mức <strong>-0.50</strong>. Nếu ở giao dịch này V14 bị <strong>giảm sâu xuống -2.97</strong>, hệ thống phát hiện đây là điểm bất thường nghiêm trọng (nghi ngờ thiết bị lạ hoặc chiếm đoạt thẻ).</li>
        </ul>
    </div>
    <div style="font-size: 0.72rem; color: #6B7280; margin-top: 8px; font-style: italic;">
        * Lưu ý: Bảng đối chiếu dưới đây giúp kiểm soát viên hiểu rõ trực quan vì sao giao dịch bị nghi ngờ, còn các chỉ số thực tế sẽ được chuẩn hóa trước khi đưa vào mô hình AI tính toán xác suất.
    </div>
</div>
<div style="background-color: #ffffff; border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-top: 10px;">
    <table style="width: 100%; border-collapse: collapse; font-size: 0.76rem; text-align: left; color: #374151;">
        <thead>
            <tr style="background-color: #F8FAFC; border-bottom: 2px solid #E5E7EB; color: #4B5563; font-weight: bold;">
                <th style="padding: 7px 8px;">Biến số</th>
                <th style="padding: 7px 8px;">Giao dịch này</th>
                <th style="padding: 7px 8px;">Mốc tham khảo</th>
                <th style="padding: 7px 8px; text-align: right;">Đánh giá</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
</div>"""

def build_probability_calculation_html(primary_algo, scaled_amt, scaled_time, prob_val):
    """Tạo phần công thức ngắn gọn giải thích cách mô hình đang chọn tính xác suất."""
    features = pd.DataFrame(
        [[scaled_amt, scaled_time] + list(st.session_state.v_array)],
        columns=MODEL_FEATURE_NAMES
    )

    if primary_algo == "Random Forest":
        model = all_assets["RF_SMOTE"]
        tree_probs = [float(tree.predict_proba(features.to_numpy())[0][1]) for tree in model.estimators_]
        preview = ", ".join(f"{value:.4f}" for value in tree_probs[:5])
        calculated_prob = float(np.mean(tree_probs))
        return f"""
<div class="pipeline-formula">
    <strong>Cách tính Random Forest:</strong><br>
    Mỗi cây trong rừng đưa ra một xác suất Fraud p<sub>i</sub> dựa trên cùng một hàng dữ liệu đầu vào.<br>
    Kết quả cuối cùng là trung bình của {len(tree_probs)} cây:<br>
    p = (p<sub>1</sub> + ... + p<sub>{len(tree_probs)}</sub>) / {len(tree_probs)}<br>
      = <strong>{calculated_prob:.6f} = {calculated_prob * 100:.2f}%</strong><br>
    <span style="font-size:0.82rem; color:#475569;"><b>Trong đó:</b> p<sub>i</sub> là xác suất Fraud của cây i.</span><br>
    5 cây đầu:
    <span style="font-size:0.82rem; color:#475569;">[{preview}, ...]</span>
</div>"""

    if primary_algo == "Decision Tree":
        model = all_assets["DT_SMOTE"]
        leaf_id = int(model.apply(features)[0])
        leaf_weights = np.asarray(model.tree_.value[leaf_id][0], dtype=float)
        normal_weight = float(leaf_weights[0])
        fraud_weight = float(leaf_weights[1])
        total_weight = normal_weight + fraud_weight
        calculated_prob = fraud_weight / total_weight if total_weight else prob_val
        return f"""
<div class="pipeline-formula">
    <strong>Cách tính Decision Tree:</strong><br>
    Dòng dữ liệu đi đến nút lá #{leaf_id}. Tại nút lá đó, xác suất Fraud được tính theo tỷ lệ:<br>
    p = Fraud / (Normal + Fraud)<br>
      = {fraud_weight:.6f} / ({normal_weight:.6f} + {fraud_weight:.6f})<br>
      = <strong>{calculated_prob:.6f} = {calculated_prob * 100:.2f}%</strong><br>
    <span style="font-size:0.82rem; color:#475569;"><b>Trong đó:</b> Fraud/Normal là số mẫu từng lớp trong nút lá.</span>
</div>"""

    if primary_algo == "Logistic Regression":
        model = all_assets["LR_SMOTE"]
        score = float(model.decision_function(features)[0])
        intercept = float(model.intercept_[0])
        contributions = features.iloc[0].to_numpy() * model.coef_[0]
        top_indices = np.argsort(np.abs(contributions))[::-1][:5]
        preview = ", ".join(f"{MODEL_FEATURE_NAMES[index]}: {contributions[index]:+.4f}" for index in top_indices)
        calculated_prob = 1.0 / (1.0 + np.exp(-score))
        return f"""
<div class="pipeline-formula">
    <strong>Cách tính Logistic Regression:</strong><br>
    Tính điểm z từ tổng trọng số của biến:<br>
    z = intercept + Σ(x<sub>i</sub> × w<sub>i</sub>)<br>
      = {intercept:.6f} + Σ(x<sub>i</sub> × w<sub>i</sub>)<br>
      = <strong>{score:.6f}</strong><br>
    <span style="font-size:0.82rem; color:#475569;"><b>Trong đó:</b> z là điểm số nội bộ, Σ(x<sub>i</sub> × w<sub>i</sub>) là tổng tích có trọng số của các biến.</span><br>
    Sau đó áp dụng sigmoid để ra xác suất:<br>
    p = 1 / (1 + e<sup>-z</sup>)<br>
      = <strong>{calculated_prob:.6f} = {calculated_prob * 100:.2f}%</strong><br>
    <span style="font-size:0.82rem; color:#475569;"><b>Trong đó:</b> 5 biến ảnh hưởng mạnh nhất: [{preview}]</span>
</div>"""

    clipped_prob = float(np.clip(prob_val, 1e-12, 1 - 1e-12))
    score = float(np.log(clipped_prob / (1 - clipped_prob)))
    return f"""
<div class="pipeline-formula">
    <strong>Cách tính 1D-CNN:</strong><br>
    30 giá trị đầu vào được xử lý qua các lớp tích chập và kết nối để sinh điểm z.<br>
    Sau đó, hàm sigmoid chuyển điểm z thành xác suất Fraud:<br>
    p = 1 / (1 + e<sup>-z</sup>)<br>
      = <strong>{prob_val:.6f} = {prob_val * 100:.2f}%</strong><br>
    <span style="font-size:0.82rem; color:#475569;"><b>Trong đó:</b> z là logit nội bộ, dùng để biểu diễn độ nghi ngờ Fraud trước khi chuẩn hóa thành xác suất.</span>
</div>"""

def render_realtime_pipeline(primary_algo, scaled_amt, scaled_time):
    prob_val = float(st.session_state.main_fraud_prob)
    threshold_val = float(st.session_state.decision_threshold_slider)
    prob_percent = f"{prob_val * 100:.2f}%"
    threshold_percent = f"{threshold_val * 100:.2f}%"
    is_fraud = st.session_state.main_prediction_label == 1
    decision_code = "FRAUD" if is_fraud else "NORMAL"
    decision_text = "CẢNH BÁO GIAN LẬN - CẦN KIỂM TRA" if is_fraud else "GIAO DỊCH HỢP LỆ - CÓ THỂ DUYỆT"
    decision_icon = "fa-triangle-exclamation" if is_fraud else "fa-circle-check"
    decision_color = "#EF4444" if is_fraud else "#10B981"
    operator = "&ge;" if is_fraud else "&lt;"
    amount_mean = float(all_assets["scaler_amount"].mean_[0])
    amount_scale = float(all_assets["scaler_amount"].scale_[0])
    time_mean = float(all_assets["scaler_time"].mean_[0])
    time_scale = float(all_assets["scaler_time"].scale_[0])
    probability_calculation = build_probability_calculation_html(primary_algo, scaled_amt, scaled_time, prob_val)

    def describe_scaled_position(value):
        if abs(value) < 0.005:
            return "gần mức trung bình lịch sử"
        direction = "cao hơn" if value > 0 else "thấp hơn"
        return f"{direction} trung bình {abs(value):.2f} lần độ dao động thường thấy"

    amount_position = describe_scaled_position(scaled_amt)
    time_position = describe_scaled_position(scaled_time)

    if prob_val >= threshold_val:
        risk_color = "#EF4444"
        risk_zone = "Vượt ngưỡng cảnh báo"
    elif prob_val >= max(threshold_val * 0.7, 0.30):
        risk_color = "#F59E0B"
        risk_zone = "Vùng xám cần lưu ý"
    else:
        risk_color = "#10B981"
        risk_zone = "Dưới ngưỡng an toàn"

    _, log_target = st.session_state.get("db_write_status", (True, "OFFLINE_FALLBACK"))
    log_target_label = "SQL Server" if log_target == "SQL_SERVER" else "Offline fallback"
    logged_txn_id = st.session_state.get("last_logged_txn_id", "TXN-NEW")
    threshold_only_recheck = st.session_state.get("threshold_only_recheck", False)
    if threshold_only_recheck:
        log_status = """<div class="pipeline-status info">
    <i class="fa-solid fa-circle-info"></i>
    <strong>Chỉ cập nhật giao diện:</strong> threshold thay đổi nên hệ thống tự tính lại quyết định,
    không ghi thêm nhật ký để tránh tạo bản ghi trùng lặp.
</div>"""
        log_code = f"Bản ghi gần nhất vẫn giữ nguyên: {logged_txn_id}"
    else:
        log_status = f"""<div class="pipeline-status">
    <i class="fa-solid fa-circle-check"></i>
    <strong>Đã lưu thành công:</strong> kết quả được ghi vào {log_target_label}.
</div>"""
        log_code = f"""INSERT INTO TransactionLogs
  (TransactionID, ModelName, Prediction, Probability)
VALUES
  ('{logged_txn_id}', '{primary_algo} - SMOTE', '{decision_code}', {prob_val:.6f});"""

    pipeline_html = f"""
<div class="pipeline-reference-head">
    <div class="pipeline-reference-title">
        <i class="fa-solid fa-code-branch"></i> Cách Hệ Thống Tính Kết Quả
    </div>
    <div class="pipeline-reference-desc">
        Theo dõi dữ liệu từ lúc tiếp nhận đến khi hệ thống đưa ra quyết định.
    </div>
</div>

<div class="pipeline-prep-flow">
    <div class="pipeline-prep-card">
        <div class="pipeline-prep-number">Chặng 1</div>
        <div class="pipeline-prep-title">Tiếp nhận dữ liệu ban đầu</div>
        <div class="pipeline-prep-note">Nhận 30 giá trị đầu vào: Amount, Time và 28 đặc trưng ẩn V1–V28.</div>
    </div>
    <div class="pipeline-prep-arrow"><i class="fa-solid fa-chevron-right"></i></div>
    <div class="pipeline-prep-card">
        <div class="pipeline-prep-number">Chặng 2</div>
        <div class="pipeline-prep-title">Chuẩn hóa Amount và Time</div>
        <div class="pipeline-prep-note">Đổi Amount và Time thành scaled_amount / scaled_time theo trung bình và độ lệch chuẩn lịch sử.</div>
    </div>
    <div class="pipeline-prep-arrow"><i class="fa-solid fa-chevron-right"></i></div>
    <div class="pipeline-prep-card">
        <div class="pipeline-prep-number">Chặng 3</div>
        <div class="pipeline-prep-title">Ghép hàng dữ liệu cuối cùng</div>
        <div class="pipeline-prep-note">Xếp scaled_amount và scaled_time trước, sau đó nối V1–V28 để tạo vector 30 chiều cho mô hình.</div>
    </div>
    <div class="pipeline-prep-arrow"><i class="fa-solid fa-chevron-right"></i></div>
    <div class="pipeline-prep-card">
        <div class="pipeline-prep-number">Chặng 4</div>
        <div class="pipeline-prep-title">Tính xác suất gian lận</div>
        <div class="pipeline-prep-note">Đưa vector 30 chiều vào mô hình SMOTE đã huấn luyện để sinh xác suất Fraud.</div>
    </div>
    <div class="pipeline-prep-arrow"><i class="fa-solid fa-chevron-right"></i></div>
    <div class="pipeline-prep-card">
        <div class="pipeline-prep-number">Chặng 5</div>
        <div class="pipeline-prep-title">Ra quyết định và lưu nhật ký</div>
        <div class="pipeline-prep-note">So sánh xác suất với ngưỡng, xác định Normal/Fraud và ghi lại kết quả.</div>
    </div>
</div>

<details class="inference-detail" open>
    <summary><i class="fa-solid fa-chevron-down"></i> Bước 1 - Nhận 30 giá trị ban đầu</summary>
    <div class="inference-detail-body">
        <div class="pipeline-status">
            <i class="fa-solid fa-circle-check"></i>
            <strong>Đầu vào đã nhận đủ:</strong> 2 giá trị quen thuộc + 28 tín hiệu ẩn danh = 30 giá trị.
        </div>
        <div class="pipeline-insight">
            <i class="fa-solid fa-lightbulb"></i>
            <strong>Hiểu đơn giản:</strong> Amount là số tiền giao dịch. Time là thời điểm giao dịch tính bằng giây.
            V1 - V28 là các tín hiệu đã ẩn danh để bảo vệ thông tin khách hàng.
        </div>
        <div class="inference-code">{{
  "Time": {st.session_state.time_val:.1f},
  "Amount": {st.session_state.amount_val:.2f},
  "V1": {st.session_state.v_array[0]:.6f},
  "V4": {st.session_state.v_array[3]:.6f},
  "V10": {st.session_state.v_array[9]:.6f},
  "V12": {st.session_state.v_array[11]:.6f},
  "V14": {st.session_state.v_array[13]:.6f},
  "...": "23 biến V còn lại"
}}</div>
        <div class="pipeline-output">
            <strong>Đầu ra bước 1:</strong> chuyển đủ 30 giá trị sang bước 2. Chỉ Amount và Time cần đổi cách đo;
            V1 - V28 được giữ nguyên.
        </div>
    </div>
</details>

<details class="inference-detail" open>
    <summary><i class="fa-solid fa-chevron-down"></i> Bước 2 - Đổi cách đo của Amount và Time</summary>
    <div class="inference-detail-body">
        <div class="pipeline-insight">
            <i class="fa-solid fa-lightbulb"></i>
            <strong>Mục đích:</strong> đổi Amount và Time về cùng cách đo để mô hình dễ so sánh.
        </div>
        <div class="inference-mini-grid">
            <div class="inference-mini-card">
                <strong>Amount ($)</strong><br>
                Hiện tại: <strong>${st.session_state.amount_val:.2f}</strong><br>
                Công thức: <em>scaled_amount</em> = (Amount - mean) / std.<br>
                Tính: ({st.session_state.amount_val:.2f} - {amount_mean:.2f}) / {amount_scale:.2f}<br>
                Kết quả: <strong>{scaled_amt:.6f}</strong> <span style="color: #64748B;">({amount_position})</span>
                <div style="margin-top: 10px; font-size: 0.78rem; color: #475569; line-height: 1.6;">
                    <strong>Chú thích:</strong>
                    <ul style="margin: 6px 0 0 18px; padding: 0; list-style: disc; font-size: 0.78rem;">
                        <li><em>Amount</em>: số tiền giao dịch thực tế.</li>
                        <li><em>mean</em>: giá trị trung bình của Amount trong dữ liệu huấn luyện.</li>
                        <li><em>std</em>: độ lệch chuẩn cho biết mức dao động của Amount.</li>
                        <li><em>scaled_amount</em>: giá trị chuẩn hóa của Amount</li>
                    </ul>
                </div>
            </div>
            <div class="inference-mini-card">
                <strong>Time (giây)</strong><br>
                Hiện tại: <strong>{st.session_state.time_val:.1f}</strong><br>
                Công thức: <em>scaled_time</em> = (Time - mean) / std.<br>
                Tính: ({st.session_state.time_val:.1f} - {time_mean:.1f}) / {time_scale:.1f}<br>
                Kết quả: <strong>{scaled_time:.6f}</strong> <span style="color: #64748B;">({time_position})</span>
                <div style="margin-top: 10px; font-size: 0.78rem; color: #475569; line-height: 1.6;">
                    <strong>Chú thích:</strong>
                    <ul style="margin: 6px 0 0 18px; padding: 0; list-style: disc; font-size: 0.78rem;">
                        <li><em>Time</em>: thời điểm giao dịch tính bằng giây kể từ giao dịch đầu tiên trong dữ liệu.</li>
                        <li><em>mean</em>: giá trị trung bình của Time trong dữ liệu huấn luyện.</li>
                        <li><em>std</em>: độ lệch chuẩn cho biết mức dao động của Time.</li>
                        <li><em>scaled_time</em>: giá trị chuẩn hóa của Time, cho biết giao dịch này xa hay gần so với thời gian trung bình.</li>
                    </ul>
                </div>
            </div>
        </div>
        <div class="pipeline-formula">
            <strong>Cách đọc:</strong> số dương = cao hơn trung bình; số âm = thấp hơn trung bình; gần 0 = gần trung bình.
        </div>
        <div class="pipeline-output">
            <strong>Đầu ra bước 2:</strong> Amount được đổi thành <strong>{scaled_amt:.6f}</strong>,
            Time được đổi thành <strong>{scaled_time:.6f}</strong>. Hai số mới này sẽ thay thế Amount và Time gốc.
        </div>
    </div>
</details>

<details class="inference-detail" open>
    <summary><i class="fa-solid fa-chevron-down"></i> Bước 3 - Ghép hàng dữ liệu sẵn sàng chấm điểm</summary>
    <div class="inference-detail-body">
        <div class="pipeline-insight">
            <i class="fa-solid fa-lightbulb"></i>
            <strong>Cách ghép:</strong> bỏ Amount và Time gốc, thay bằng 2 giá trị mới từ bước 2.
            Sau đó đặt V1 - V28 ở phía sau theo đúng thứ tự.
        </div>
        <div class="inference-code">[scaled_amount, scaled_time, V1, V2, ..., V28]
[{scaled_amt:.6f}, {scaled_time:.6f}, {st.session_state.v_array[0]:.6f}, {st.session_state.v_array[1]:.6f}, ..., {st.session_state.v_array[27]:.6f}]</div>
        <div class="pipeline-output">
            <strong>Đầu ra bước 3:</strong> một hàng gồm đúng 30 số, sẵn sàng chuyển sang bước 4 để tính xác suất rủi ro.
        </div>
    </div>
</details>

<details class="inference-detail" open>
    <summary><i class="fa-solid fa-chevron-down"></i> Bước 4 - Tính xác suất rủi ro và đưa ra quyết định</summary>
    <div class="inference-detail-body">
        <div class="pipeline-insight">
            <i class="fa-solid fa-lightbulb"></i>
            <strong>Dữ liệu đưa vào:</strong> <strong>{primary_algo} - SMOTE</strong> nhận hàng 30 giá trị ở bước 3.
            SMOTE chỉ được dùng lúc huấn luyện, không tạo dữ liệu mới khi kiểm tra giao dịch này.
        </div>
        {probability_calculation}
        <div class="pipeline-probability">Xác suất rủi ro gian lận:</div>
        <div class="inference-risk-track">
            <div class="inference-risk-fill" style="width: {prob_val * 100:.2f}%; background: {risk_color};"></div>
        </div>
        <div class="pipeline-probability-value" style="color: {risk_color};">{prob_percent}</div>
        <div style="text-align: center; color: #64748B; font-size: 0.7rem;">{risk_zone}</div>
        <div class="pipeline-insight">
            <i class="fa-solid fa-lightbulb"></i>
            <strong>Ra quyết định:</strong> so xác suất vừa tính với ngưỡng cảnh báo ở khung bên trái.
        </div>
        <div class="inference-decision-grid">
            <div class="inference-metric-box">
                <div class="inference-metric-label">Xác suất tính toán</div>
                <div class="inference-metric-value" style="color: {risk_color};">{prob_percent}</div>
            </div>
            <div class="inference-operator">{operator}</div>
            <div class="inference-metric-box threshold">
                <div class="inference-metric-label">Ngưỡng hệ thống</div>
                <div class="inference-metric-value" style="color: #7C3AED;">{threshold_percent}</div>
            </div>
        </div>
        <div class="inference-final" style="background: {decision_color};">
            <i class="fa-solid {decision_icon}"></i> {decision_text}
        </div>
        <div class="pipeline-formula">
            Quy tắc: nếu p &ge; ngưỡng thì cảnh báo Fraud.<br>
            Hiện tại: <strong>{prob_percent} {operator} {threshold_percent}</strong> → <strong>{decision_code}</strong>
        </div>
    </div>
</details>

<details class="inference-detail" open>
    <summary><i class="fa-solid fa-chevron-down"></i> Bước 5 - Lưu kết quả để tra cứu</summary>
    <div class="inference-detail-body">
        <div class="pipeline-insight">
            <i class="fa-solid fa-lightbulb"></i>
            <strong>Lưu lại:</strong> mã giao dịch, mô hình, kết quả và xác suất.
        </div>
        {log_status}
        <div class="inference-code">{log_code}</div>
    </div>
</details>
"""
    st.html(pipeline_html)

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

    return assets

# Nạp tài nguyên vào bộ nhớ
try:
    all_assets = load_all_assets()
except Exception as e:
    st.error("Lỗi nghiêm trọng khi tải mô hình học máy. Vui lòng kiểm tra lại thư mục models/.")
    st.exception(e)
    st.stop()

@st.cache_data(show_spinner="Đang phân tích đóng góp các biến số (SHAP)...")
def get_shap_explanation_cached(primary_algo, model_type, features_tuple):
    import shap

    # 1. Khởi tạo dataframe
    features_arr = np.array(features_tuple).reshape(1, -1)
    df_temp = pd.DataFrame(features_arr, columns=MODEL_FEATURE_NAMES)

    # Lấy mô hình từ assets
    assets = load_all_assets()

    main_model = None
    if model_type == "tree":
        if "Random Forest" in primary_algo:
            main_model = assets["RF_SMOTE"]
        else:
            main_model = assets["DT_SMOTE"]
    elif model_type == "linear":
        main_model = assets["LR_SMOTE"]
    elif model_type == "cnn":
        main_model = assets["CNN_SMOTE"]

    shap_values = None
    base_val = 0.5

    if model_type == "tree":
        explainer = shap.TreeExplainer(main_model)
        shap_values_raw = explainer.shap_values(df_temp)
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        elif len(shap_values_raw.shape) == 3:
            shap_values = shap_values_raw[:, :, 1]
        else:
            shap_values = shap_values_raw
        base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

    elif model_type == "linear":
        background = np.zeros((1, 30))
        def lr_predict_fn(x):
            return main_model.predict_proba(x)[:, 1]
        explainer = shap.KernelExplainer(lr_predict_fn, background)
        shap_values_raw = explainer.shap_values(df_temp.values)
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[0]
        else:
            shap_values = shap_values_raw
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

    elif model_type == "cnn":
        background = np.zeros((1, 30))
        def cnn_predict_fn(x):
            x_3d = x.reshape(x.shape[0], x.shape[1], 1)
            return main_model.predict(x_3d, verbose=0).flatten()
        explainer = shap.KernelExplainer(cnn_predict_fn, background)
        shap_values_raw = explainer.shap_values(df_temp.values)
        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[0]
        else:
            shap_values = shap_values_raw
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[0]

    return shap_values, base_val

# =====================================================================
# 5. GIỮ VÀ ĐỒNG BỘ SESSION STATE CHO SCENARIO VÀ SLIDERS
# =====================================================================
if "scenario_select" not in st.session_state:
    st.session_state.scenario_select = "Tự cấu hình thông số (Custom)"

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
    st.session_state.processed_df_main = None
    st.session_state.pending_input_changes = False

# Khởi tạo bổ sung đề phòng hot-reload từ phiên bản cũ thiếu trường V4
if "v4_val" not in st.session_state:
    st.session_state.v4_val = 5.22

# Hàm chuyển đổi sang chế độ Custom khi có bất cứ tham số nào bị thay đổi
def on_parameter_change():
    st.session_state.scenario_select = "Tự cấu hình thông số (Custom)"

NORMAL_SCENARIO = "TXN-001 (Mẫu Giao dịch Hợp lệ - Normal)"
FRAUD_SCENARIO = "TXN-007 (Mẫu Giao dịch Nghi ngờ Gian lận - Fraud)"
GRAY_ZONE_SCENARIO = "Giao dịch nghi ngờ (Vùng xám)"
REAL_DATA_SCENARIOS = {NORMAL_SCENARIO, FRAUD_SCENARIO, GRAY_ZONE_SCENARIO}

# Hàm đồng bộ hóa khi chuyển đổi giao dịch mẫu
def on_scenario_change():
    scen = st.session_state.get("scenario_select")
    if not scen:
        return
    if scen == NORMAL_SCENARIO:
        # Dòng thật raw index 150513 trong creditcard.csv, thuộc X_test, Class = 0.
        # RandomForest_SMOTE.predict_proba trả xác suất rủi ro đúng 6.00%.
        st.session_state.time_val = 93581.0
        st.session_state.amount_val = 444.98
        st.session_state.v14_val = 2.525498
        st.session_state.v4_val = 1.376933
        st.session_state.v12_val = -3.338335
        st.session_state.v10_val = -0.054195
        st.session_state.v_array = [-1.922986, -0.504946, -1.201207, 1.376933, -0.029364, 0.169823, 3.022871, -0.819448, 0.50209, -0.054195, 1.33928, -3.338335, -0.236799, 2.525498, -0.537372, -0.258999, 0.217676, 0.868803, 0.96718, -1.519464, -0.346873, 0.768965, 0.919211, -0.444387, 0.114991, -0.366782, 0.538271, -0.229909]
    elif scen == FRAUD_SCENARIO:
        # Dòng thật raw index 18809 trong creditcard.csv, thuộc X_test, Class = 1.
        # RandomForest_SMOTE.predict_proba trả xác suất rủi ro đúng 74.00%.
        st.session_state.time_val = 29785.0
        st.session_state.amount_val = 30.30
        st.session_state.v14_val = -2.971317
        st.session_state.v4_val = 1.72168
        st.session_state.v12_val = -2.549177
        st.session_state.v10_val = -2.895252
        st.session_state.v_array = [0.923764, 0.344048, -2.880004, 1.72168, -3.019565, -0.639736, -3.801325, 1.299096, 0.864065, -2.895252, 3.028162, -2.549177, -1.560432, -2.971317, 1.078895, -4.702012, -4.908099, -1.508873, 3.001685, 0.170872, 0.899931, 1.481271, 0.725266, 0.17696, -1.815638, -0.536517, 0.489035, -0.049729]
    elif scen == GRAY_ZONE_SCENARIO:
        # Dòng thật raw index 10204 trong creditcard.csv, thuộc X_test.
        # RandomForest_SMOTE.predict_proba trả xác suất rủi ro đúng 40.00%.
        st.session_state.time_val = 15817.0
        st.session_state.amount_val = 11.39
        st.session_state.v14_val = -3.26647554459315
        st.session_state.v4_val = 2.507298518227
        st.session_state.v12_val = -6.54261034532574
        st.session_state.v10_val = 1.07741752106048
        st.session_state.v_array = [-4.64189285087538, 2.90208643306647, -1.57293870931742, 2.507298518227, -0.871782564349351, -1.0409025722626, -1.59390073966645, -3.25490508581579, 1.90896268593707, 1.07741752106048, 3.33850216040272, -6.54261034532574, 1.09953638182128, -3.26647554459315, 1.01472790444294, -4.84238307827835, -5.269876299726, -2.34466857415508, 1.95922405409272, -0.465679220344122, 1.96359666634146, -0.217413915973245, -0.54933995583883, 0.645545202036521, -0.354557818794591, -0.611763845006479, -3.90808047547799, -0.671248265147117]
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

    if scen in REAL_DATA_SCENARIOS:
        st.session_state.v_string_input_area = ", ".join([repr(float(v)) for v in st.session_state.v_array])
    else:
        st.session_state.v_string_input_area = ", ".join([f"{v:.4f}" for v in st.session_state.v_array])
    for i in range(28):
        st.session_state[f"v_field_{i}"] = float(st.session_state.v_array[i])

# Header (Full width, but very small height)
st.markdown("""
    <div style="text-align: center; margin-top: -15px; margin-bottom: 25px;">
        <h2 style="color: #0A2540; font-size: 1.8rem; font-weight: 800; display: inline-flex; align-items: center; gap: 10px; margin-bottom: 5px;">
            <i class="fa-solid fa-shield-halved" style="color: #1E3A8A; font-size: 1.6rem;"></i> HỆ THỐNG PHÂN TÍCH & PHÁT HIỆN GIAN LẬN THẺ TÍN DỤNG
        </h2>
    </div>
""", unsafe_allow_html=True)

# Khởi tạo hai cột chính: Cột trái (Điều khiển) và Cột phải (Kết quả/Phân tích)
col_left, col_right = st.columns([1, 1.25], gap="medium")

with col_left:
    st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: -20px; margin-bottom: 5px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px;'><i class='fa-solid fa-sliders'></i> Cấu Hình Chung</h4>", unsafe_allow_html=True)

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
            [FRAUD_SCENARIO, GRAY_ZONE_SCENARIO, NORMAL_SCENARIO, "Tự cấu hình thông số (Custom)"],
            key="scenario_select",
            on_change=on_scenario_change
        )

    decision_threshold = st.slider(
        "Ngưỡng quyết định (Threshold):",
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        key="decision_threshold_slider"
    )

    st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: -20px; margin-bottom: 5px; border-bottom: 1px solid #E5E7EB; padding-bottom: 3px;'><i class='fa-solid fa-wand-magic-sparkles'></i> Tinh Chỉnh What-If (Các biến tác động mạnh)</h4>", unsafe_allow_html=True)

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

    # Các slider what-if là nguồn dữ liệu cuối cùng cho bốn biến mạnh.
    # Đồng bộ sau bước parse chuỗi để giá trị textarea cũ không ghi đè thay đổi vừa kéo.
    final_v_inputs = list(final_v_inputs)
    final_v_inputs[13] = v14_val
    final_v_inputs[3] = v4_val
    final_v_inputs[11] = v12_val
    final_v_inputs[9] = v10_val
    st.session_state.v_array = final_v_inputs

    predict_clicked = st.button("KIỂM TRA GIAO DỊCH VÀ GHI NHẬT KÝ VẾT", type="primary", use_container_width=True)

# Chuẩn hóa dữ liệu thô đầu vào để chạy mô hình
scaled_amt = all_assets["scaler_amount"].transform([[st.session_state.amount_val]])[0][0]
scaled_time = all_assets["scaler_time"].transform([[st.session_state.time_val]])[0][0]
features_array = np.array([scaled_amt, scaled_time] + final_v_inputs).reshape(1, -1)

# Tự động theo dõi sự thay đổi của tham số.
# Threshold chỉ thay đổi quyết định, nên sau lần kiểm tra đầu tiên có thể tự tính lại
# mà không ghi thêm nhật ký. Các đầu vào khác vẫn yêu cầu người dùng bấm kiểm tra lại.
input_params_changed = False
threshold_changed = False
if "prev_time_val" in st.session_state and st.session_state.prev_time_val != st.session_state.time_val:
    input_params_changed = True
if "prev_amount_val" in st.session_state and st.session_state.prev_amount_val != st.session_state.amount_val:
    input_params_changed = True
if "prev_v_array" in st.session_state and st.session_state.prev_v_array != st.session_state.v_array:
    input_params_changed = True
if "prev_primary_algo" in st.session_state and st.session_state.prev_primary_algo != primary_algo:
    input_params_changed = True
if "prev_decision_threshold" in st.session_state and st.session_state.prev_decision_threshold != decision_threshold:
    threshold_changed = True
if "prev_input_method" in st.session_state and st.session_state.prev_input_method != input_method:
    input_params_changed = True

st.session_state.prev_time_val = st.session_state.time_val
st.session_state.prev_amount_val = st.session_state.amount_val
st.session_state.prev_v_array = st.session_state.v_array.copy()
st.session_state.prev_primary_algo = primary_algo
st.session_state.prev_decision_threshold = decision_threshold
st.session_state.prev_input_method = input_method

if input_params_changed:
    st.session_state.pending_input_changes = True
    st.session_state.run_prediction = False
    st.session_state.threshold_only_recheck = False

auto_recheck_threshold = (
    threshold_changed
    and st.session_state.run_prediction
    and not st.session_state.get("pending_input_changes", False)
)

if predict_clicked or auto_recheck_threshold:
    st.session_state.run_prediction = True
    if predict_clicked:
        st.session_state.pending_input_changes = False
        st.session_state.refresh_logs = True
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
        if primary_algo == "Random Forest" and cfg["key"] == "RF_SMOTE":
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

    # Tạo dataframe biểu diễn 30 cột đầy đủ phục vụ SHAP
    processed_df_main = pd.DataFrame(features_array, columns=MODEL_FEATURE_NAMES)

    main_duration = int((time.time() - start_time) * 1000)

    # Lưu thông tin vào session để dùng khi reload lại giao diện Streamlit
    st.session_state.results_table = results_table
    st.session_state.main_fraud_prob = main_fraud_prob
    st.session_state.main_prediction_label = main_prediction_label
    st.session_state.processed_df_main = processed_df_main
    st.session_state.threshold_only_recheck = auto_recheck_threshold

    if predict_clicked:
        # GHI VÀO SQL SERVER / LOCAL FALLBACK
        try:
            existing_logs, _, _ = fetch_all_logs()
            if not isinstance(existing_logs, list):
                existing_logs = []
        except Exception:
            existing_logs = []

        if "TXN-007" in selected_txn:
            matching_ids = [log for log in existing_logs if log.get("TransactionID") and "TXN-007" in str(log.get("TransactionID"))]
            next_num = len(matching_ids) + 1
            txn_id_log = f"TXN-007-{next_num:03d}"
        elif "TXN-001" in selected_txn:
            matching_ids = [log for log in existing_logs if log.get("TransactionID") and "TXN-001" in str(log.get("TransactionID"))]
            next_num = len(matching_ids) + 1
            txn_id_log = f"TXN-001-{next_num:03d}"
        elif selected_txn == GRAY_ZONE_SCENARIO:
            matching_ids = [log for log in existing_logs if log.get("TransactionID") and "TXN-GRAY" in str(log.get("TransactionID"))]
            next_num = len(matching_ids) + 1
            txn_id_log = f"TXN-GRAY-{next_num:03d}"
        else:
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
        st.session_state.last_logged_txn_id = txn_id_log

# =====================================================================
# 6. ĐIỀU HƯỚNG TAB NGANG TRÊN CÙNG & HIỂN THỊ CHI TIẾT
# =====================================================================
selected_detail_id = None

with col_right:
    tab_shap, tab_process, tab_logs = st.tabs([
        "KẾT QUẢ & GIẢI THÍCH (SHAP)",
        "CÁCH HỆ THỐNG XỬ LÝ",
        "NHẬT KÝ LƯU VẾT"
    ])

    with tab_shap:
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
            shap_values = None
            base_val = 0.5
            feature_impact = None
            top_5_features = None
            shap_error = None

            # Lấy model đang được chọn để giải thích SHAP động
            model_type = None
            if "Random Forest" in primary_algo:
                model_type = "tree"
            elif primary_algo == "Decision Tree":
                model_type = "tree"
            elif primary_algo == "Logistic Regression":
                model_type = "linear"
            elif primary_algo == "1D-CNN (Deep Learning)":
                model_type = "cnn"

            cur_processed_df = st.session_state.processed_df_main

            try:
                features_tuple = tuple(cur_processed_df.values[0])
                shap_values, base_val = get_shap_explanation_cached(primary_algo, model_type, features_tuple)

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
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 8px; margin-bottom: 4px;'><i class='fa-solid fa-bolt'></i> Các lực đang tác động đến kết quả (SHAP Force Plot)</h4>", unsafe_allow_html=True)

            if shap_error is None and shap_values is not None:
                st_shap(shap.force_plot(base_val, shap_values[0], cur_processed_df), height=135)
                st.markdown(f"""
                    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 8px; color: #78350F; font-size: 0.78rem; line-height: 1.5; margin-top: 4px; margin-bottom: 12px; padding: 10px 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                        <strong style="color: #D97706; display: flex; align-items: center; gap: 6px; font-size: 0.82rem; margin-bottom: 4px;">
                            <i class="fa-solid fa-bolt"></i> Hướng dẫn đọc biểu đồ lực tác động (SHAP Force Plot):
                        </strong>
                        <ul style="margin: 4px 0 0 16px; padding: 0; list-style-type: disc;">
                            <li><strong>Mức xuất phát (Base Value):</strong> Xác suất rủi ro trung bình ban đầu (quanh mốc {base_val:.2f}) trước khi xét các thuộc tính của giao dịch này.</li>
                            <li><strong>Lực đẩy màu đỏ (<span style="color:#DC2626; font-weight:bold;">higher</span>):</strong> Các chỉ số bất thường đang ra sức <strong>đẩy</strong> xác suất rủi ro gian lận lên cao hơn.</li>
                            <li><strong>Lực kéo màu xanh (<span style="color:#2563EB; font-weight:bold;">lower</span>):</strong> Các chỉ số ổn định đang ra sức <strong>níu</strong> rủi ro về mức thấp an toàn.</li>
                            <li><strong>Xác suất cuối cùng (f(x) - màu đậm):</strong> Kết quả sau khi cộng trừ các lực kéo co, ở đây là <strong>{cur_prob:.2%}</strong>.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='height: 135px; background: #F9FAFB; border-radius:6px; display:flex; align-items:center; justify-content:center; color:#9CA3AF; font-size:0.8rem; margin-bottom:8px;'>SHAP Force Plot không khả dụng</div>", unsafe_allow_html=True)

            # --- SHAP RANKING ---
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 8px; margin-bottom: 4px;'><i class='fa-solid fa-chart-bar'></i> 5 biến ảnh hưởng nhiều nhất đến giao dịch này</h4>", unsafe_allow_html=True)

            if shap_error is not None:
                st.warning(f"Không thể tải SHAP. Chi tiết: {shap_error}")
            elif shap_values is not None and top_5_features is not None:
                render_shap_feature_explanations(top_5_features)
            else:
                st.info("Không có dữ liệu SHAP.")

            # --- BẢNG SO SÁNH ---
            st.markdown("<h4 style='color: #0A2540; font-size: 0.95rem; font-weight: 700; margin-top: 8px; margin-bottom: 6px;'><i class='fa-solid fa-code-compare'></i> Đối Chiếu Với Mốc Tham Khảo Minh Họa</h4>", unsafe_allow_html=True)
            st.markdown(build_reference_comparison_html(), unsafe_allow_html=True)

    with tab_process:
        st.markdown("<h3 style='color: #0A2540; font-size: 1.3rem; font-weight: 800; margin-bottom: 15px;'><i class='fa-solid fa-gears'></i> Cách Hệ Thống Xử Lý Giao Dịch</h3>", unsafe_allow_html=True)

        sys_tab1, sys_tab2 = st.tabs([
            "CÁCH TÍNH KẾT QUẢ",
            "SO SÁNH 8 MÔ HÌNH"
        ])

        with sys_tab1:
            if not st.session_state.run_prediction:
                st.markdown("""
                    <div class="inference-empty">
                        <div style="font-size: 2rem; margin-bottom: 6px;"><i class="fa-solid fa-diagram-project"></i></div>
                        <strong>Chưa có kết quả xử lý để hiển thị</strong><br>
                        Chọn giao dịch ở khung bên trái và bấm <strong>KIỂM TRA GIAO DỊCH VÀ GHI NHẬT KÝ VẾT</strong>.<br>
                        Hệ thống sẽ mô phỏng trực quan từng bước bằng chính dữ liệu vừa phân tích.
                    </div>
                """, unsafe_allow_html=True)
            else:
                render_realtime_pipeline(primary_algo, scaled_amt, scaled_time)

        with sys_tab2:
            st.markdown("<h4 style='color: #0A2540; font-size: 1rem; font-weight: 700; margin-top: 10px; margin-bottom: 5px;'><i class='fa-solid fa-table-list'></i> So Sánh Kết Quả Của 8 Mô Hình</h4>", unsafe_allow_html=True)
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

                    fig, ax = plt.subplots(figsize=(5.2, 2.45))
                    bars = ax.barh(names, probs, color=colors, height=0.42)
                    ax.set_xlabel('Xác suất rủi ro (%)', fontsize=8, color='#374151')
                    ax.set_xlim(0, 118)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#D1D5DB')
                    ax.spines['bottom'].set_color('#D1D5DB')
                    ax.tick_params(axis='y', which='major', labelsize=7.5, colors='#374151')
                    ax.tick_params(axis='x', which='major', labelsize=7, colors='#374151')

                    # Thêm nhãn số phần trăm ngay sau mỗi cột
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 1.0, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                                va='center', ha='left', fontsize=7, color='#374151', fontweight='bold')

                    plt.tight_layout(pad=0.65)
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                except Exception as ex:
                    st.warning(f"Không thể tải đồ thị trực quan. Chi tiết: {str(ex)}")

    with tab_logs:
        if "logs_cache" not in st.session_state or st.session_state.get("refresh_logs", False):
            all_logs_data, conn_mode, _ = fetch_all_logs()
            st.session_state.logs_cache = all_logs_data
            st.session_state.logs_conn_mode = conn_mode
            st.session_state.refresh_logs = False
        else:
            all_logs_data = st.session_state.logs_cache
            conn_mode = st.session_state.logs_conn_mode

        available_log_dates = []
        for log_item in all_logs_data:
            parsed_timestamp = pd.to_datetime(log_item.get("Timestamp"), errors="coerce")
            if not pd.isna(parsed_timestamp):
                available_log_dates.append(parsed_timestamp.date())

        today = datetime.date.today()
        default_start_date = min(available_log_dates) if available_log_dates else today
        default_end_date = max(available_log_dates) if available_log_dates else today
        if "applied_log_start_date" not in st.session_state:
            st.session_state.applied_log_start_date = default_start_date
        if "applied_log_end_date" not in st.session_state:
            st.session_state.applied_log_end_date = default_end_date

        col_t3_header, col_filter_start, col_filter_end, col_filter_btn, col_t3_btn = st.columns(
            [3.1, 1.55, 1.55, 0.72, 0.85],
            vertical_alignment="bottom"
        )
        with col_t3_header:
            st.markdown("<h4 style='color: #0A2540; font-size: 1rem; font-weight: 700; margin-top: 5px; margin-bottom: 5px;'><i class='fa-solid fa-database'></i> Nhật ký Giám sát</h4>", unsafe_allow_html=True)
        with col_filter_start:
            selected_log_start_date = st.date_input(
                "Từ ngày",
                value=st.session_state.applied_log_start_date,
                format="DD/MM/YYYY",
                label_visibility="collapsed",
                key="log_filter_start"
            )
        with col_filter_end:
            selected_log_end_date = st.date_input(
                "Đến ngày",
                value=st.session_state.applied_log_end_date,
                format="DD/MM/YYYY",
                label_visibility="collapsed",
                key="log_filter_end"
            )
        with col_filter_btn:
            apply_log_filter = st.button("LỌC", key="log_filter_apply", use_container_width=True)
        with col_t3_btn:
            if st.button("Tải lại", key="log_refresh", use_container_width=True):
                st.session_state.refresh_logs = True
                st.rerun()

        invalid_log_date_range = selected_log_start_date > selected_log_end_date
        if apply_log_filter and not invalid_log_date_range:
            st.session_state.applied_log_start_date = selected_log_start_date
            st.session_state.applied_log_end_date = selected_log_end_date

        if invalid_log_date_range:
            applied_start_date = selected_log_start_date
            applied_end_date = selected_log_end_date
            st.warning("Khoảng ngày không hợp lệ: Từ ngày phải nhỏ hơn hoặc bằng Đến ngày. Vui lòng chọn lại.")
        else:
            applied_start_date = st.session_state.applied_log_start_date
            applied_end_date = st.session_state.applied_log_end_date

        logs_data = []
        for log_item in all_logs_data:
            parsed_timestamp = pd.to_datetime(log_item.get("Timestamp"), errors="coerce")
            if not pd.isna(parsed_timestamp) and applied_start_date <= parsed_timestamp.date() <= applied_end_date:
                logs_data.append(log_item)

        if not invalid_log_date_range and conn_mode == "SQL_SERVER":
            st.markdown("""
                <div style="display: inline-flex; align-items: center; gap: 6px; background-color: #DEF7EC; border: 1px solid #31C48D; color: #03543F; padding: 4px 8px; border-radius: 50px; font-size: 0.75rem; font-weight: bold; margin-top: -12px; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 6px; height: 6px; background-color: #31C48D; border-radius: 50%;"></span>
                     ĐÃ KẾT NỐI SQL SERVER
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="display: inline-flex; align-items: center; gap: 6px; background-color: #FDF2F2; border: 1px solid #F8B4B4; color: #9B1C1C; padding: 4px 8px; border-radius: 50px; font-size: 0.75rem; font-weight: bold; margin-top: -12px; margin-bottom: 4px;">
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

            # 2. Vẽ 2 đồ thị so sánh (ở giữa)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("<h5 style='color: #0A2540; font-size: 0.85rem; font-weight: 700; margin-bottom: 5px; text-align: center;'><i class='fa-solid fa-chart-pie'></i> Tỷ Lệ Kết Quả Phân Loại</h5>", unsafe_allow_html=True)
                try:
                    # Biểu đồ tròn Donut so sánh loại Giao dịch (Fraud vs Normal)
                    fig1, ax1 = plt.subplots(figsize=(4.5, 2.3))
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
                    fig2, ax2 = plt.subplots(figsize=(4.5, 2.3))

                    model_counts = df_raw["ModelName"].value_counts()
                    # Rút ngắn tên mô hình để tránh chồng chữ.
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
            st.markdown("""
                <p style="font-size: 0.78rem; color: #2563EB; margin: -2px 0 8px 0; font-style: italic;">
                    <i class="fa-solid fa-circle-info"></i>
                    Nhấp vào bất kỳ dòng nào dưới đây để xem dữ liệu thô tương ứng trong bảng TransactionLog_Detail.
                </p>
            """, unsafe_allow_html=True)

            selected_detail_id = render_clickable_logs_table(df_logs)

        elif not invalid_log_date_range:
            st.info(
                f"Không có nhật ký giao dịch từ {applied_start_date:%d/%m/%Y} "
                f"đến {applied_end_date:%d/%m/%Y}."
            )

if selected_detail_id:
    show_detail_modal(selected_detail_id)
