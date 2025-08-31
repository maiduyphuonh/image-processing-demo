import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ======================
# Config + CSS
# ======================
st.set_page_config(page_title="Image Processing Demo", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #ffffff;
    }
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em !important;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f0f8ff;
    }
    footer {
        visibility: hidden;
    }
    .custom-footer {
        text-align: center;
        padding: 10px;
        margin-top: 20px;
        color: #0066cc;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Tiêu đề
# ======================
st.title("📸 Image Processing Demo")

# ======================
# Sidebar controls
# ======================
st.sidebar.header("⚙️ Cài đặt")

uploaded_file = st.sidebar.file_uploader("👉 Upload ảnh", type=["jpg", "jpeg", "png"])

option = st.sidebar.selectbox("🔧 Chọn phép biến đổi", 
                      ["Original", "Negative", "Log Transform", "Gamma Correction", 
                       "Piecewise-linear", "Histogram Equalization", "CLAHE"])

show_hist = st.sidebar.checkbox("📊 Hiển thị Histogram")

# ======================
# Xử lý ảnh
# ======================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    result_img = img_rgb.copy()

    if option == "Negative":
        result_img = 255 - img_rgb

    elif option == "Log Transform":
        c_val = 255 / np.log(1 + np.max(img_gray))
        log_img = c_val * (np.log1p(img_gray.astype(float)))
        log_img = np.clip(log_img, 0, 255).astype(np.uint8)
        result_img = cv2.cvtColor(log_img, cv2.COLOR_GRAY2RGB)

    elif option == "Gamma Correction":
        gamma = st.sidebar.slider("Gamma", 0.1, 5.0, 1.0)
        gamma_img = np.array(255 * (img_rgb / 255) ** gamma, dtype=np.uint8)
        result_img = gamma_img

    elif option == "Piecewise-linear":
        r1 = st.sidebar.slider("r1", 0, 255, 70)
        s1 = st.sidebar.slider("s1", 0, 255, 0)
        r2 = st.sidebar.slider("r2", 0, 255, 140)
        s2 = st.sidebar.slider("s2", 0, 255, 255)

        def pixel_val(p):
            if p < r1:
                return (s1 / r1) * p
            elif p < r2:
                return ((s2 - s1) / (r2 - r1)) * (p - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (p - r2) + s2

        img_out = np.array([pixel_val(p) for p in img_gray.flatten()],
                           dtype=np.uint8).reshape(img_gray.shape)
        result_img = cv2.cvtColor(img_out, cv2.COLOR_GRAY2RGB)

    elif option == "Histogram Equalization":
        img_eq = cv2.equalizeHist(img_gray)
        result_img = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)

    elif option == "CLAHE":
        clip = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0)
        tile = st.sidebar.slider("Tile Grid Size", 2, 16, 8)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        img_clahe = clahe.apply(img_gray)
        result_img = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)

    # ======================
    # Hiển thị ảnh
    # ======================
    col1, col2 = st.columns(2)
    col1.image(img_rgb, caption="Ảnh gốc", use_container_width=True)
    col2.image(result_img, caption=f"Kết quả - {option}", use_container_width=True)

    # ======================
    # Histogram
    # ======================
    if show_hist:
        st.subheader("📊 Histogram so sánh")
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(img_gray.ravel(), bins=256, range=[0,256], color='blue', alpha=0.7)
        ax[0].set_title("Ảnh gốc")
        ax[1].hist(cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY).ravel(),
                   bins=256, range=[0,256], color='green', alpha=0.7)
        ax[1].set_title("Ảnh sau xử lý")
        st.pyplot(fig)

# ======================
# Footer
# ======================
st.markdown('<div class="custom-footer">💙 Powered by OpenCV + Streamlit</div>',
            unsafe_allow_html=True)
