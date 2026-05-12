import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="Görüntü Sahteciliği Tespit Sistemi",
    page_icon="🕵️",
    layout="wide"
)
IMG_SIZE = 224

@st.cache_resource
def load_ai_models():
    cnn_model = load_model("models/cnn_model.h5")
    lstm_model = load_model("models/lstm_model.h5")
    return cnn_model, lstm_model

cnn_model, lstm_model = load_ai_models()

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
    color: #1f2937;
}

.block-container {
    padding-top: 2rem;
    max-width: 1100px;
    margin: auto;
}

html, body, [class*="css"], [class*="stMarkdown"], p, span, div, label {
    font-size: 14px;
    color: #1f2937 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #111827 !important;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #374151 !important;
    margin-bottom: 30px;
}

.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 15px;
    border: 1px solid #dbe3ef;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
}

.card h3,
.card p {
    color: #1f2937 !important;
}

.metric-box {
    background-color: #ffffff;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #dbe3ef;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
}

.metric-title {
    color: #6b7280 !important;
    font-size: 15px;
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #111827 !important;
}

.result-success {
    background-color: #e8f8ef;
    padding: 18px;
    border-radius: 16px;
    color: #166534 !important;
    font-size: 22px;
    font-weight: bold;
    border: 1px solid #bbf7d0;
}

.result-warning {
    background-color: #fff7df;
    padding: 18px;
    border-radius: 16px;
    color: #92400e !important;
    font-size: 22px;
    font-weight: bold;
    border: 1px solid #fde68a;
}

.result-danger {
    background-color: #fee2e2;
    padding: 18px;
    border-radius: 16px;
    color: #991b1b !important;
    font-size: 22px;
    font-weight: bold;
    border: 1px solid #fecaca;
}


button:hover {
    background-color: #f3f4f6 !important;
}
            
/* File uploader alanı */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #dbe3ef;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
}

/* Drag drop kutusu */
[data-testid="stFileUploaderDropzone"] {
    background-color: #eef4ff !important;
    border: 2px dashed #7aa7ff !important;
    border-radius: 16px !important;
    padding: 25px !important;
}

/* File uploader iç yazılar */
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #1f2937 !important;
}

/* Browse files butonu */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}

/* Buton hover */
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #1d4ed8 !important;
    color: white !important;
}
            
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📌 Proje Bilgileri")
st.sidebar.info(
    "Bu yazılım, görüntüler üzerinde sahtecilik izlerini tespit etmek için "
    "ORB, AKAZE, SIFT ve AI tabanlı analiz yöntemlerini kullanır."
)

st.sidebar.markdown("### Kullanılan Yöntemler")
st.sidebar.write("✅ ORB")
st.sidebar.write("✅ AKAZE")
st.sidebar.write("✅ SIFT")
st.sidebar.write("✅ Şüpheli Bölge Tespiti")
st.sidebar.write("✅ AI Risk Analizi")

st.markdown(
    '<div class="main-title">🕵️ Görüntü Sahteciliği Tespit Sistemi</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Bu sistem, yüklenen görüntülerde sahtecilik izlerini tespit etmek için görüntü işleme ve yapay zeka tabanlı analiz yöntemleri kullanır.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "📤 Analiz edilecek görüntüyü yükleyin",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    st.markdown("## 📷 Yüklenen Görüntü")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(image, caption="Analiz edilen görüntü", width=350)

    st.markdown("## 🔍 Algoritma Analizleri")

    tab1, tab2, tab3, tab4 = st.tabs(["ORB", "AKAZE", "SIFT", "SURF"])

    with tab1:
        orb = cv2.ORB_create()
        kp_orb, des_orb = orb.detectAndCompute(gray, None)
        img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 255, 0))

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.image(img_orb, caption="ORB Keypoints", width=350)

        st.success(f"ORB Keypoint Sayısı: {len(kp_orb)}")

    with tab2:
        akaze = cv2.AKAZE_create()
        kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
        img_akaze = cv2.drawKeypoints(img, kp_akaze, None, color=(255, 0, 0))

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.image(img_akaze, caption="AKAZE Keypoints", width=350)

        st.success(f"AKAZE Keypoint Sayısı: {len(kp_akaze)}")

    with tab3:
        try:
            sift = cv2.SIFT_create()
            kp_sift, des_sift = sift.detectAndCompute(gray, None)
            img_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 0, 255))

            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                st.image(img_sift, caption="SIFT Keypoints", width=350)

            st.success(f"SIFT Keypoint Sayısı: {len(kp_sift)}")

        except Exception:
            st.warning("SIFT bu OpenCV sürümünde çalışmıyor.")

    with tab4:
        try:
            surf = cv2.xfeatures2d.SURF_create(400)
            kp_surf, des_surf = surf.detectAndCompute(gray, None)
            img_surf = cv2.drawKeypoints(img, kp_surf, None, color=(255, 255, 0))

            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                st.image(img_surf, caption="SURF Keypoints", width=350)

            st.success(f"SURF Keypoint Sayısı: {len(kp_surf)}")

        except Exception:
            st.warning("SURF algoritması lisans kısıtları nedeniyle bu OpenCV sürümünde çalıştırılamıyor.")

    st.markdown("## 🚨 Şüpheli Bölge Analizi")

    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    marked_img = img.copy()
    suspicious_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            suspicious_count += 1

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(marked_img, caption="Şüpheli bölgeler", width=350)

    if suspicious_count > 5:
        st.warning("Görüntüde çok sayıda şüpheli bölge tespit edildi.")
    else:
        st.success("Görüntüde belirgin bir sahtecilik izi görülmedi.")

    st.markdown("## 🤖 AI ile Sahtecilik Tespiti")

    ai_image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    ai_image = preprocess_input(ai_image.astype(np.float32))

    cnn_input = np.expand_dims(ai_image, axis=0)
    lstm_input = np.expand_dims(ai_image, axis=0)

    cnn_prediction = cnn_model.predict(cnn_input, verbose=0)
    lstm_prediction = lstm_model.predict(lstm_input, verbose=0)

    cnn_real_score = cnn_prediction[0][0] * 100
    cnn_fake_score = cnn_prediction[0][1] * 100

    lstm_real_score = lstm_prediction[0][0] * 100
    lstm_fake_score = lstm_prediction[0][1] * 100

    final_score = (cnn_fake_score * 0.7) + (lstm_fake_score * 0.3)

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">CNN Sahtecilik Skoru</div>
        <div class="metric-value">{cnn_fake_score:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">LSTM Sahtecilik Skoru</div>
        <div class="metric-value">{lstm_fake_score:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Nihai Risk Skoru</div>
        <div class="metric-value">{final_score:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚦 Nihai Sonuç")

    if final_score > 20:
        st.markdown(
            '<div class="result-danger">🔴 Sonuç: Görüntü sahte/manipüle edilmiş olabilir.</div>',
            unsafe_allow_html=True
        )
    elif final_score > 5:
        st.markdown(
            '<div class="result-warning">🟡 Sonuç: Görüntü şüpheli görünüyor.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-success">🟢 Sonuç: Görüntü orijinal olabilir.</div>',
            unsafe_allow_html=True
        )


    st.markdown("## 📄 Rapor Özeti")

    st.info(
        "Bu raporda görüntü; ORB, AKAZE, SIFT, şüpheli bölge analizi ve AI risk skoru ile incelenmiştir. "
        "Sonuçlar kesin adli kanıt niteliğinde değildir; görüntü sahteciliği ihtimaline yönelik teknik bir ön değerlendirme sunar."
    )

else:
    st.markdown("""
    <div class="card">
        <h3>🧭 Kullanım Adımları</h3>
        <p>1. Görüntü yükleyin.</p>
        <p>2. Sistem görüntüyü otomatik analiz eder.</p>
        <p>3. ORB, AKAZE, SIFT ve AI sonuçlarını inceleyin.</p>
        <p>4. Genel sonucu rapor olarak değerlendirin.</p>
    </div>
    """, unsafe_allow_html=True)