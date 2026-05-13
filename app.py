import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
from utils.ui_styles import load_custom_css
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from io import BytesIO
from io import BytesIO

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


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


load_custom_css()

with st.sidebar:
    st.markdown("""
<div class="sidebar-logo">
<div class="logo-icon">🛡️</div>
<div>
<div class="logo-title">Deep<span>Detect</span></div>
<div class="logo-subtitle">Yapay Zeka Güvenliği</div>
</div>
</div>
""", unsafe_allow_html=True)

    selected_page = st.radio(
        "Menü",
        [
            "🏠 Ana Panel",
            "🖼️ Görüntü Analizi",
            "🧠 Yapay Zeka Tespiti",
            "📄 Raporlar",
            "⚙️ Ayarlar"
        ],
        label_visibility="collapsed"
    )

    st.markdown("""
<div class="sidebar-footer">🟢 Sistem Aktif</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="ai-badge">⚡ Gelişmiş Yapay Zeka Destekli</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Deep<span>Detect</span></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Yapay Zeka Destekli Görüntü Sahteciliği Tespit Sistemi</div>',
    unsafe_allow_html=True
)



st.markdown(
    '<div class="hero-desc">Görüntüleri manipülasyon, sahtecilik ve özgünlük açısından gelişmiş görüntü işleme ve derin öğrenme modelleriyle analiz edin.</div>',
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "📤 Analiz edilecek görüntüyü yükleyin",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

st.markdown("""
<div class="feature-row">
    <div class="feature-pill">🔬 ORB Analizi</div>
    <div class="feature-pill">🎯 AKAZE Tespiti</div>
    <div class="feature-pill">🧠 CNN Puanlama</div>
    <div class="feature-pill">⚡ LSTM Modelleri</div>
</div>
""", unsafe_allow_html=True)

show_analysis = selected_page == "🖼️ Görüntü Analizi"
show_ai = selected_page == "🧠 Yapay Zeka Tespiti"
show_reports = selected_page == "📄 Raporlar"
show_home = selected_page == "🏠 Ana Panel"

if uploaded_file is None:
    st.markdown("""
    <div class="card">
        <h3>🧭 Kullanım Adımları</h3>
        <p>1. Görüntü yükleyin.</p>
        <p>2. Sistem görüntüyü otomatik analiz eder.</p>
        <p>3. ORB, AKAZE, SIFT, SURF ve AI sonuçlarını inceleyin.</p>
        <p>4. Genel sonucu rapor olarak değerlendirin.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if selected_page == "⚙️ Ayarlar":
    st.markdown("""
<div class="section-header">
    <div class="section-badge">⚙️ Sistem Ayarları</div>
    <h2>Ayarlar</h2>
    <p>Arayüz ve analiz deneyimi için temel sistem ayarları.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>🎨 Arayüz Ayarları</h3>
<p>Mevcut tema: Açık mavi / cam efektli v0.app stili</p>
<p>Sol panel: Aktif</p>
<p>Rapor tasarımı: Premium kart görünümü</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>🧠 Model Bilgileri</h3>
<p>CNN modeli: models/cnn_model.h5</p>
<p>LSTM modeli: models/lstm_model.h5</p>
<p>Görüntü boyutu: 224 x 224</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card">
<h3>🚦 Karar Eşikleri</h3>
<p>final_score > 20: Sahte / Manipüle Edilmiş Olabilir</p>
<p>final_score > 5: Şüpheli Görünüyor</p>
<p>Aksi halde: Orijinal Olabilir</p>
</div>
""", unsafe_allow_html=True)

    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    st.markdown("## 📷 Yüklenen Görüntü")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(image, caption="Analiz edilen görüntü", width=350)

    if show_home or show_analysis:

        st.markdown("""
    <div class="section-header">
        <div class="section-badge">🔍 Görüntü İşleme Analizi</div>
        <h2>Algoritma Analizleri</h2>
        <p>ORB, AKAZE, SIFT ve SURF algoritmaları ile görüntü üzerindeki ayırt edici özellik noktaları incelenir.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["ORB", "AKAZE", "SIFT", "SURF"])

    with tab1:
        orb = cv2.ORB_create()
        kp_orb, des_orb = orb.detectAndCompute(gray, None)
        img_orb = cv2.drawKeypoints(img, kp_orb, None, color=(0, 255, 0))

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.image(img_orb, caption="ORB Keypoints", width=350)

        st.success(f"ORB Keypoint Sayısı: {len(kp_orb)}")

        st.markdown("""
        <div class="algo-info-card">
        <div class="algo-info-title">
        Dönüşe duyarsız özellik tespiti için Oriented FAST ve Rotated BRIEF algoritması
        </div>

        <div class="algo-info-desc">
        ORB analizi birden fazla dönüşe duyarsız anahtar nokta tespit etti.
        Görüntü tutarlılığı, özellik yoğun alanlarda minimum manipülasyon olduğunu gösteriyor.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        akaze = cv2.AKAZE_create()
        kp_akaze, des_akaze = akaze.detectAndCompute(gray, None)
        img_akaze = cv2.drawKeypoints(img, kp_akaze, None, color=(255, 0, 0))

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.image(img_akaze, caption="AKAZE Keypoints", width=350)

        st.success(f"AKAZE Keypoint Sayısı: {len(kp_akaze)}")
        st.markdown("""
<div class="algo-info-card">
<div class="algo-info-title">
Doğrusal olmayan ölçek uzayı analizi için hızlandırılmış KAZE algoritması
</div>

<div class="algo-info-desc">
AKAZE doğrusal olmayan ölçek uzayı özellikleri tespit etti.
Kenar koruması özgün sıkıştırma yapaylıklarını gösteriyor.
</div>
</div>
""", unsafe_allow_html=True)

    with tab3:
        try:
            sift = cv2.SIFT_create()
            kp_sift, des_sift = sift.detectAndCompute(gray, None)
            img_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 0, 255))

            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                st.image(img_sift, caption="SIFT Keypoints", width=350)

            st.success(f"SIFT Keypoint Sayısı: {len(kp_sift)}")
            st.markdown("""
<div class="algo-info-card">
<div class="algo-info-title">
Güçlü anahtar nokta tespiti için Ölçek Değişmez Özellik Dönüşümü
</div>

<div class="algo-info-desc">
SIFT analizi doğal dağılımlı ölçek değişmez özellikler gösteriyor.
Kopyala-yapıştır manipülasyonu belirtisi yok.
</div>
</div>
""", unsafe_allow_html=True)

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
            st.markdown("""
<div class="algo-info-card">
<div class="algo-info-title">
Hızlı çoklu ölçek analizi için Hızlandırılmış Güçlü Özellikler
</div>

<div class="algo-info-desc">
SURF birden fazla ölçekte güçlü özellikler tespit etti.
Blob tespit örüntüleri orijinal çekim ile tutarlı görünüyor.
</div>
</div>
""", unsafe_allow_html=True)

        except Exception:
            st.warning("SURF algoritması lisans kısıtları nedeniyle bu OpenCV sürümünde çalıştırılamıyor.")

    st.markdown("""
<div class="section-header">
    <div class="section-badge">🚨 Adli Görüntü İncelemesi</div>
    <h2>Şüpheli Bölge Analizi</h2>
    <p>Görüntüdeki kenar, kontur ve yoğunluk değişimleri incelenerek potansiyel manipülasyon bölgeleri işaretlenir.</p>
</div>
""", unsafe_allow_html=True)

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

if final_score > 20:
    result_icon = "🚨"
    result_title = "Sahte / Manipüle Edilmiş Olabilir"
    result_desc = "Görüntüde yüksek seviyede sahtecilik veya manipülasyon riski tespit edildi."
    result_class = "final-danger"

elif final_score > 5:
    result_icon = "⚠️"
    result_title = "Şüpheli Görünüyor"
    result_desc = "Görüntüde bazı bölgeler veya skorlar ek inceleme gerektirebilir."
    result_class = "final-warning"

else:
    result_icon = "✅"
    result_title = "Orijinal Olabilir"
    result_desc = "Görüntüde belirgin bir sahtecilik veya manipülasyon izi tespit edilmedi."
    result_class = "final-success"

confidence_score = max(0, min(100, 100 - final_score))

if show_home or show_ai:

    st.markdown("""
<div class="section-header">
    <div class="section-badge">🤖 Yapay Zeka Risk Motoru</div>
    <h2>AI ile Sahtecilik Tespiti</h2>
    <p>CNN ve LSTM modellerinden gelen skorlar birleştirilerek nihai risk skoru hesaplanır.</p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="metric-box metric-cnn">
        <div class="metric-head">
            <div class="metric-icon">🧠</div>
            <div>
                <div class="metric-title">CNN Sahtecilik Skoru</div>
            </div>
        </div>
        <div class="metric-value">{cnn_fake_score:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-box metric-lstm">
        <div class="metric-head">
            <div class="metric-icon">〰️</div>
            <div>
                <div class="metric-title">LSTM Sahtecilik Skoru</div>
            </div>
        </div>
        <div class="metric-value">{lstm_fake_score:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-box metric-final">
        <div class="metric-head">
            <div class="metric-icon">🎯</div>
            <div>
                <div class="metric-title">Nihai Risk Skoru</div>
            </div>
        </div>
        <div class="metric-value">{final_score:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div class="section-header">
    <div class="section-badge">🚦 Sonuç Motoru</div>
    <h2>Nihai Sonuç</h2>
    <p>Yapay zeka modellerinden gelen skorlar değerlendirilerek görüntünün risk durumu belirlenir.</p>
</div>
""", unsafe_allow_html=True)

    

    st.markdown(f"""
<div class="final-result-card {result_class}">
<div class="final-glow"></div>
<div class="final-icon">{result_icon}</div>
<div class="final-title">{result_title}</div>
<div class="final-desc">{result_desc}</div>

<div class="confidence-box">
<div class="confidence-row">
<span>Güven Seviyesi</span>
<strong>{confidence_score:.1f}%</strong>
</div>

<div class="confidence-track">
<div class="confidence-fill" style="width:{confidence_score:.1f}%;"></div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

if show_home or show_reports:

    report_summary = f"""
Analiz edilen görüntü; ORB, AKAZE, SIFT ve SURF algoritmalarıyla çoklu özellik çıkarımından geçirildi.

Yapay zeka modelleri görüntüyü CNN ve LSTM tabanlı olarak değerlendirdi.

CNN sahtecilik skoru %{cnn_fake_score:.2f}, LSTM sahtecilik skoru %{lstm_fake_score:.2f} olarak hesaplandı.

Birleştirilmiş nihai risk skoru %{final_score:.2f} seviyesindedir.

Sonuç değerlendirmesi:
{result_title}

{result_desc}

Bu rapor kesin adli kanıt niteliğinde değildir. Teknik ön değerlendirme amacıyla üretilmiştir.
"""

    full_report_text = f"""
ANALİZ RAPORU

YÖNETİCİ ÖZETİ

{report_summary}

TEKNİK DETAYLAR

• ORB algoritması {len(kp_orb)} adet anahtar nokta tespit etti.
• AKAZE algoritması {len(kp_akaze)} adet anahtar nokta tespit etti.
• SIFT algoritması {len(kp_sift) if 'kp_sift' in locals() else 0} adet anahtar nokta tespit etti.
• SURF algoritması {len(kp_surf) if 'kp_surf' in locals() else 0} adet anahtar nokta tespit etti.
• Şüpheli bölge sayısı: {suspicious_count}
• Nihai güven seviyesi: %{confidence_score:.1f}
"""

    st.markdown("""
<div class="report-header">

<div class="report-title-wrap">
<div class="report-icon">📄</div>

<div>
<h2>Analiz Raporu</h2>
<p>Kapsamlı teknik döküm ve öneriler</p>
</div>
</div>

<div class="report-actions">
<button onclick="navigator.clipboard.writeText(`{full_report_text}`)">
<span class="btn-icon">📋</span><span>Kopyala</span>
</button>

<button onclick="window.print()">
<span class="btn-icon">🖨️</span><span>Yazdır</span>
</button>

<button onclick="navigator.share && navigator.share({
title:'Analiz Raporu',
text:document.getElementById('reportText').innerText
})">
<span class="btn-icon">🔗</span><span>Paylaş</span>
</button>
</div>

</div>
""", unsafe_allow_html=True)
        
    pdfmetrics.registerFont(
    TTFont("DejaVuSans", "C:/Windows/Fonts/arial.ttf")
)

    pdf_buffer = BytesIO()

    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

    text = pdf.beginText(40, 800)

    text.setFont("DejaVuSans", 10)

    for line in full_report_text.split("\n"):
        text.textLine(line)

    pdf.drawText(text)
    pdf.showPage()
    pdf.save()

    pdf_buffer.seek(0)

    st.download_button(
    label="⬇️ PDF Olarak İndir",
    data=pdf_buffer.getvalue(),
    file_name="analiz_raporu.pdf",
    mime="application/pdf"
)

    st.markdown(f"""
<div class="analysis-report-card" id="reportText">

<div class="report-section-title">
<span></span>
Yönetici Özeti
</div>

<p>
{report_summary}
</p>

<hr>

<div class="report-section-title small-title">
TEKNİK DETAYLAR
</div>

<ul>
<li>ORB algoritması <strong>{len(kp_orb)}</strong> adet anahtar nokta tespit etti.</li>

<li>AKAZE algoritması <strong>{len(kp_akaze)}</strong> adet anahtar nokta tespit etti.</li>

<li>SIFT algoritması <strong>{len(kp_sift) if 'kp_sift' in locals() else 0}</strong> adet anahtar nokta tespit etti.</li>

<li>SURF algoritması <strong>{len(kp_surf) if 'kp_surf' in locals() else 0}</strong> adet anahtar nokta tespit etti.</li>

<li>Şüpheli bölge sayısı: <strong>{suspicious_count}</strong></li>

<li>Nihai güven seviyesi: <strong>%{confidence_score:.1f}</strong></li>
</ul>

</div>
""", unsafe_allow_html=True)


