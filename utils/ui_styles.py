import streamlit as st

def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        color-scheme: light;
        --bg: #f8fbff;
        --panel: rgba(255,255,255,0.72);
        --panel-strong: rgba(255,255,255,0.86);
        --panel-border: rgba(56,189,248,0.15);
        --text: #0f172a;
        --muted: #475569;
        --accent: #06b6d4;
        --accent-strong: #0891b2;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
    }

    * {
        box-sizing: border-box;
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 18% 12%, rgba(6,182,212,0.18), transparent 24%),
                    radial-gradient(circle at 85% 10%, rgba(14,165,233,0.14), transparent 20%),
                    linear-gradient(135deg, #f8fbff 0%, #eef7fb 42%, #f9fcff 100%);
        color: var(--text);
    }

    .main-title {
        text-align: center;
        font-size: 72px;
        font-weight: 900;
        letter-spacing: -1px;
        margin: 0 auto 0.75rem auto;
        color: var(--text) !important;
    }

    .main-title span {
        color: var(--accent-strong);
        text-shadow: 0 0 26px rgba(6,182,212,0.28);
    }

    .subtitle {
        text-align: center;
        font-size: 1.45rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
        color: #334155 !important;
    }

    .hero-desc {
        max-width: 720px;
        margin: 0 auto 2rem auto;
        color: #64748b !important;
        font-size: 1rem;
        line-height: 1.75;
    }

    .ai-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.8rem auto;
        padding: 0.85rem 1.6rem;
        border-radius: 999px;
        font-weight: 700;
        color: var(--accent-strong) !important;
        background: rgba(207,250,254,0.8);
        border: 1px solid rgba(6,182,212,0.24);
        box-shadow: 0 16px 40px rgba(6,182,212,0.12);
    }

    .card, .metric-box, .glass-card {
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(148,163,184,0.18);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-radius: 28px;
        box-shadow: 0 20px 55px rgba(15,23,42,0.08);
    }

    .card {
        padding: 28px;
        margin-top: 2rem;
    }

    .feature-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 0.85rem;
        margin: 2rem auto 2.5rem auto;
        max-width: 940px;
    }

    .feature-pill {
        padding: 0.95rem 1.25rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        color: #475569 !important;
        font-weight: 700;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        z-index: 1;
    }

    .feature-pill:hover {
        transform: translateY(-3px);
        border-color: rgba(6,182,212,0.38);
        box-shadow: 0 20px 45px rgba(6,182,212,0.16);
    }

    [data-testid="stFileUploader"] {
        max-width: 860px !important;
        margin: 0 auto 2rem auto;
        background: rgba(255,255,255,0.82) !important;
        border: 1px solid rgba(148,163,184,0.22) !important;
        border-radius: 30px !important;
        padding: 28px !important;
        box-shadow: 0 30px 90px rgba(15,23,42,0.12) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 270px !important;
        border-radius: 28px !important;
        border: 2px dashed rgba(6,182,212,0.35) !important;
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(246,253,255,0.9) 100%) !important;
    }

    /* ===== V0 STYLE TABS ===== */

.stTabs [data-baseweb="tab-list"] {
    gap: 14px;
    background: rgba(255,255,255,0.58);
    padding: 6px;
    border-radius: 22px;
    width: fit-content;
    margin: 0 auto 28px auto;

    border: 1px solid rgba(255,255,255,0.8);

    box-shadow:
        0 10px 35px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.9);

    backdrop-filter: blur(18px);
}

button[data-baseweb="tab"] {
    height: 42px !important;

    padding: 0 18px !important;

    border-radius: 15px !important;

    background: transparent !important;

    color: #111827 !important;

    font-size: 16px !important;
    font-weight: 700 !important;

    border: none !important;

    transition: all .22s ease;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    gap: 8px;
}

/* Hover */

button[data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.55) !important;
}

/* Active */

button[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(34,211,238,0.18) !important;

    color: #0891b2 !important;

    box-shadow:
        0 6px 18px rgba(6,182,212,0.14);

    border: 1px solid rgba(34,211,238,0.28) !important;
}

/* Alt kırmızı çizgiyi kaldır */

button[data-baseweb="tab"]::after {
    display: none !important;
}
    .section-header {
        margin: 4rem auto 1.8rem auto;
        text-align: center;
        max-width: 780px;
    }

    .section-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem 1.4rem;
        border-radius: 999px;
        background: rgba(207,250,254,0.82);
        border: 1px solid rgba(6,182,212,0.20);
        color: var(--accent-strong) !important;
        font-weight: 800;
        letter-spacing: 0.03em;
        box-shadow: 0 14px 35px rgba(6,182,212,0.12);
        margin-bottom: 1rem;
    }

    .section-header h2 {
        font-size: 2.4rem;
        font-weight: 900;
        margin: 0;
        color: var(--text) !important;
    }

    .section-header p {
        margin: 1rem auto 0 auto;
        color: #64748b !important;
        font-size: 1rem;
        line-height: 1.75;
    }

    div[data-testid="stImage"] {
    background: rgba(255,255,255,0.82);
    padding: 18px 18px 22px 18px !important;

    border-radius: 34px;

    border: 1px solid rgba(255,255,255,0.9);

    box-shadow:
        0 18px 50px rgba(15,23,42,0.08),
        inset 0 1px 0 rgba(255,255,255,0.9);

    backdrop-filter: blur(16px);

    width: fit-content !important;
    min-width: 430px !important;
    max-width: 100% !important;

    margin: 0 auto 28px auto;

    transition: all .25s ease;

    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
}

div[data-testid="stImage"] img {
    border-radius: 24px !important;
    display: block !important;
}
    

    div[data-testid="stImage"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 28px 80px rgba(6,182,212,0.18);
    }

    .stAlert {
        border-radius: 20px !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        box-shadow: 0 16px 45px rgba(15,23,42,0.06) !important;
        font-size: 0.98rem !important;
    }

    .final-result-card {
        position: relative;
        overflow: hidden;
        margin: 2.5rem auto 2rem auto;
        padding: 38px 28px;
        border-radius: 32px;
        text-align: center;
        background: rgba(255,255,255,0.78);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        border: 1px solid rgba(148,163,184,0.20);
        box-shadow: 0 32px 95px rgba(15,23,42,0.10);
    }

    .final-glow {
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0.45;
    }

    .final-success .final-glow {
        background: radial-gradient(circle at top, rgba(34,197,94,0.24), transparent 55%);
    }

    .final-warning .final-glow {
        background: radial-gradient(circle at top, rgba(245,158,11,0.26), transparent 55%);
    }

    .final-danger .final-glow {
        background: radial-gradient(circle at top, rgba(239,68,68,0.28), transparent 55%);
    }

    .final-icon {
        position: relative;
        width: 98px;
        height: 98px;
        margin: 0 auto 18px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.1rem;
        background: rgba(255,255,255,0.88);
        box-shadow: 0 24px 70px rgba(15,23,42,0.12);
        z-index: 1;
    }

    .final-title {
        position: relative;
        font-size: 2rem;
        font-weight: 900;
        color: var(--text) !important;
        margin-bottom: 0.75rem;
        z-index: 1;
    }

    .final-desc {
        position: relative;
        color: #64748b !important;
        font-size: 1rem;
        line-height: 1.8;
        max-width: 720px;
        margin: 0 auto 1.8rem auto;
        z-index: 1;
    }

    .confidence-box {
        position: relative;
        max-width: 720px;
        margin: 0 auto;
        padding: 20px 24px;
        border-radius: 24px;
        background: rgba(248,250,252,0.88);
        border: 1px solid rgba(148,163,184,0.18);
        z-index: 1;
    }

    .confidence-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        gap: 1rem;
        color: #334155 !important;
        font-weight: 700;
        flex-wrap: wrap;
    }

    .confidence-row strong {
        color: var(--accent-strong) !important;
        font-size: 1.1rem;
    }

    .confidence-track {
        height: 14px;
        border-radius: 999px;
        background: rgba(203,213,225,0.78);
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #06b6d4, #0891b2);
    }

    h1, h2, h3 {
        color: var(--text) !important;
    }

    p, span, div, label {
        color: #334155 !important;
    }

    .stButton>button {
        border-radius: 18px !important;
        font-weight: 700 !important;
        padding: 0.9rem 1.6rem !important;
        background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 18px 45px rgba(6,182,212,0.22) !important;
    }

    .stButton>button:hover {
        opacity: 0.96 !important;
    }
                

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 28px 20px !important;
}
                
/* V0 SIDEBAR */

.v0-sidebar{
    display:flex;
    flex-direction:column;
    height:100vh;
    padding:10px 8px;
}

.sidebar-logo{
    display:flex;
    align-items:center;
    gap:14px;
    padding:20px 14px 30px 14px;
    border-bottom:1px solid rgba(255,255,255,0.08);
}

.logo-icon{
    width:56px;
    height:56px;
    border-radius:18px;
    background:linear-gradient(135deg,#06b6d4,#0891b2);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:28px;
    box-shadow:0 10px 30px rgba(6,182,212,0.35);
}

.logo-title{
    font-size:34px;
    font-weight:800;
    color:white !important;
    line-height:1;
}

.logo-title span{
    color:#22d3ee !important;
}

.logo-subtitle{
    color:#94a3b8 !important;
    font-size:14px;
    margin-top:6px;
}

.sidebar-menu{
    padding-top:24px;
    display:flex;
    flex-direction:column;
    gap:10px;
}

.menu-item{
    padding:18px 18px;
    border-radius:18px;
    color:#cbd5e1 !important;
    font-size:18px;
    font-weight:600;
    transition:0.25s;
    border:1px solid transparent;
}

.menu-item:hover{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(34,211,238,0.25);
}

.menu-item.active{
    background:rgba(34,211,238,0.15);
    color:#22d3ee !important;
    border:1px solid rgba(34,211,238,0.35);
}

.menu-item.settings{
    margin-top:10px;
}

.sidebar-footer{
    margin-top:auto;
    padding:16px;
    border-radius:18px;
    background:rgba(255,255,255,0.04);
    color:#4ade80 !important;
    text-align:center;
    font-weight:700;
}           

.sidebar-logo {
    gap: 12px;
    padding: 18px 10px 28px 10px;
}

.logo-icon {
    width: 48px;
    height: 48px;
    min-width: 48px;
    font-size: 24px;
    border-radius: 16px;
}

.logo-title {
    font-size: 28px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.logo-subtitle {
    font-size: 13px !important;
    white-space: nowrap;
}

.menu-item {
    font-size: 16px !important;
    padding: 16px 18px !important;
    line-height: 1.4;
}

.sidebar-menu {
    gap: 8px !important;
}
                
                /* ===== V0 HERO FIX ===== */

.main-title {
    font-size: 82px !important;
    line-height: 1 !important;
    letter-spacing: -3px;
    margin-bottom: 10px !important;
    font-weight: 900 !important;
}

.main-title span {
    color: #06b6d4 !important;
    text-shadow:
        0 0 18px rgba(6,182,212,.25),
        0 0 40px rgba(6,182,212,.18);
}

.subtitle {
    font-size: 22px !important;
    font-weight: 700 !important;
    margin-top: 12px !important;
    margin-bottom: 18px !important;
    color: #334155 !important;
}

.hero-desc {
    max-width: 640px;
    margin: auto;
    line-height: 1.7;
    color: #64748b !important;
    font-size: 17px;
}

/* Upload Card */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.72) !important;
    border: 1px solid rgba(255,255,255,.8) !important;
    backdrop-filter: blur(18px);
    border-radius: 30px !important;
    padding: 18px !important;
    margin-top: 30px;
    box-shadow:
        0 10px 40px rgba(15,23,42,.05),
        inset 0 1px 0 rgba(255,255,255,.7);
}

[data-testid="stFileUploaderDropzone"] {
    min-height: 260px !important;
    border-radius: 26px !important;
    border: 2px dashed rgba(103,232,249,.8) !important;
    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,.65),
            rgba(255,255,255,.45)
        ) !important;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Upload text */

[data-testid="stFileUploaderDropzone"] div {
    color: #334155 !important;
}

/* Pills */

.feature-row {
    gap: 18px !important;
    margin-top: 24px !important;
    margin-bottom: 55px !important;
}

.feature-pill {
    padding: 14px 24px !important;
    border-radius: 999px !important;
    background: rgba(255,255,255,.75) !important;
    border: 1px solid rgba(203,213,225,.6) !important;
    box-shadow: 0 6px 18px rgba(15,23,42,.04);
    font-weight: 600;
    color: #475569 !important;
}

/* Background glow */

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at top center,
        rgba(34,211,238,.16),
        transparent 40%);
    pointer-events: none;
    z-index: 0;
}
/* Badge hizalama düzeltmesi */
.ai-badge {
    display: flex !important;
    width: fit-content !important;
    margin: 16px auto 32px auto !important;
    transform: none !important;
    position: relative !important;
    top: 0 !important;
    z-index: 2 !important;
}
                
/* ===== V0 UPLOAD AREA FULL MATCH ===== */

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(24px);
    border-radius: 34px !important;

    padding: 14px !important;
    margin-top: 20px !important;

    box-shadow:
        0 20px 60px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.95) !important;
}

/* Inner upload area */

[data-testid="stFileUploaderDropzone"] {
    min-height: 220px !important;

    border-radius: 30px !important;

    border: 2px dashed rgba(103,232,249,0.85) !important;

    background:
        linear-gradient(
            180deg,
            rgba(255,255,255,0.82),
            rgba(255,255,255,0.62)
        ) !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    transition: all .25s ease;
}

/* Hover */

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #22d3ee !important;

    box-shadow:
        0 0 0 4px rgba(34,211,238,0.08),
        0 20px 40px rgba(34,211,238,0.10);
}

/* Upload content center */

[data-testid="stFileUploaderDropzone"] section {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Upload icon */

[data-testid="stFileUploaderDropzone"] svg {
    width: 82px !important;
    height: 82px !important;

    padding: 18px;
    border-radius: 26px;

    background: linear-gradient(
        180deg,
        rgba(207,250,254,1),
        rgba(224,242,254,0.95)
    );

    color: #0891b2 !important;

    box-shadow:
        0 12px 30px rgba(6,182,212,0.16);
}

/* Main text */

[data-testid="stFileUploaderDropzone"] h3,
[data-testid="stFileUploaderDropzone"] div {
    color: #0f172a !important;
}

/* Upload title */

[data-testid="stFileUploaderDropzone"] div {
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* Small text */

[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #64748b !important;
    font-size: 15px !important;
}

/* Browse button */

[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(
        135deg,
        #06b6d4,
        #0891b2
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 18px !important;

    padding: 14px 28px !important;

    font-size: 16px !important;
    font-weight: 700 !important;

    box-shadow:
        0 12px 30px rgba(6,182,212,0.25);

    transition: all .25s ease;
}

[data-testid="stBaseButton-secondary"]:hover {
    transform: translateY(-2px);

    box-shadow:
        0 18px 40px rgba(6,182,212,0.32);
}

                
/* ===== ALGORITHM INFO CARD ===== */

.algo-info-card{
    margin-top:18px;

    background:rgba(255,255,255,0.72);

    border:1px solid rgba(203,213,225,0.55);

    border-radius:24px;

    padding:20px 22px;

    box-shadow:
        0 10px 30px rgba(15,23,42,0.05);

    backdrop-filter:blur(14px);
}

.algo-info-title{
    font-size:15px;
    font-weight:600;

    color:#64748b !important;

    line-height:1.6;

    margin-bottom:12px;
}

.algo-info-desc{
    font-size:17px;
    line-height:1.8;

    color:#0f172a !important;

    font-weight:500;
}
                
.metric-head {
    position: relative;
    display: flex;
    align-items: center;
    gap: 14px;
    z-index: 2;
}

.metric-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;

    display: flex;
    align-items: center;
    justify-content: center;

    color: white !important;
    font-size: 24px;

    background: linear-gradient(135deg, #06b6d4, #2563eb);
    box-shadow: 0 14px 32px rgba(6,182,212,0.25);
}

.metric-risk {
    color: #d97706 !important;
    font-size: 14px;
    font-weight: 700;
    margin-top: 4px;
}

.metric-cnn::before {
    background: radial-gradient(circle, rgba(14,165,233,0.18), transparent 65%);
}

.metric-lstm::before {
    background: radial-gradient(circle, rgba(217,70,239,0.18), transparent 65%);
}

.metric-final::before {
    background: radial-gradient(circle, rgba(249,115,22,0.20), transparent 65%);
}
                
/* ===== ANALİZ RAPORU ===== */

.report-header {
    margin-top: 55px;
    margin-bottom: 24px;

    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;

    flex-wrap: wrap;
}

.report-title-wrap {
    display: flex;
    align-items: center;
    gap: 14px;
}

.report-icon {
    width: 46px;
    height: 46px;
    border-radius: 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    background: rgba(34,211,238,0.18);
    color: #0891b2 !important;

    font-size: 22px;
}

.report-header h2 {
    margin: 0;
    font-size: 34px;
    font-weight: 900;
    color: #0f172a !important;
}

.report-header p {
    margin: 6px 0 0 0;
    color: #64748b !important;
    font-size: 16px;
}

.report-actions {
    margin-top: 4px;

    display: flex;
    align-items: center;

    gap: 12px;

    flex-wrap: nowrap;
}

.report-actions button {
    height: 44px;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    gap: 8px !important;

    padding: 0 18px !important;

    border-radius: 14px !important;

    background: rgba(255,255,255,0.86) !important;

    border: 1px solid rgba(226,232,240,0.85) !important;

    color: #0f172a !important;

    font-size: 14px !important;
    font-weight: 800 !important;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.95);

    transition: all .22s ease;

    line-height: 1 !important;
    vertical-align: middle !important;
}

[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg,#06b6d4,#0891b2) !important;
    color: white !important;
    border-radius: 14px !important;
    border: none !important;
    font-weight: 800 !important;
    padding: 12px 20px !important;
    box-shadow: 0 14px 35px rgba(6,182,212,0.22) !important;
}

.analysis-report-card {
    margin-top: 24px;

    padding: 34px 36px;

    border-radius: 28px;

    background: rgba(255,255,255,0.78);

    border: 1px solid rgba(255,255,255,0.9);

    box-shadow:
        0 24px 70px rgba(15,23,42,0.08),
        inset 0 1px 0 rgba(255,255,255,0.95);

    backdrop-filter: blur(22px);

    color: #475569 !important;
}

.report-section-title {
    display: flex;
    align-items: center;
    gap: 10px;

    font-size: 20px;
    font-weight: 900;

    color: #0f172a !important;

    margin-bottom: 24px;
}

.report-section-title span {
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: #0891b2;
}

.analysis-report-card p {
    font-size: 17px;
    line-height: 1.85;
    color: #475569 !important;
    white-space: pre-line;
}

.analysis-report-card hr {
    border: none;
    border-top: 1px solid rgba(148,163,184,0.22);
    margin: 28px 0;
}

.small-title {
    font-size: 15px;
    letter-spacing: 0.06em;
}

.analysis-report-card ul {
    padding-left: 0;
    margin: 0;
    list-style: none;
}

.analysis-report-card li {
    position: relative;

    padding-left: 24px;
    margin-bottom: 14px;

    font-size: 16px;
    line-height: 1.7;

    color: #475569 !important;
}

.analysis-report-card li::before {
    content: "";

    position: absolute;
    left: 0;
    top: 11px;

    width: 7px;
    height: 7px;

    border-radius: 999px;
    background: #0891b2;
}

/* ===== PREMIUM ANALİZ RAPORU REVIZE ===== */

.report-header {
    max-width: 1100px;
    margin: 70px auto 26px auto !important;
    padding: 0 4px;

    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
}

.report-title-wrap {
    display: flex;
    align-items: center;
    gap: 16px;
}

.report-icon {
    width: 54px !important;
    height: 54px !important;
    border-radius: 18px !important;

    background: linear-gradient(135deg, rgba(207,250,254,1), rgba(186,230,253,0.95)) !important;
    color: #0891b2 !important;

    box-shadow:
        0 14px 35px rgba(6,182,212,0.16),
        inset 0 1px 0 rgba(255,255,255,0.9);
}

.report-header h2 {
    font-size: 38px !important;
    line-height: 1 !important;
    letter-spacing: -1px;
}

.report-header p {
    font-size: 15px !important;
    margin-top: 12px !important;
}

.report-actions {
    margin-top: 4px;

    display: flex;
    align-items: center;

    gap: 12px;

    flex-wrap: nowrap;
}

.report-actions button {
    height: 44px;
    padding: 0 18px !important;
    border-radius: 14px !important;

    background: rgba(255,255,255,0.86) !important;
    border: 1px solid rgba(226,232,240,0.85) !important;

    color: #0f172a !important;
    font-size: 14px !important;
    font-weight: 800 !important;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.95);

    transition: all .22s ease;
    
    line-height: 1 !important;
    
}          

.report-actions button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 36px rgba(6,182,212,0.12);
}

[data-testid="stDownloadButton"] {
    max-width: 1100px;
    margin: 0 auto 26px auto;
}

[data-testid="stDownloadButton"] button {
    height: 48px !important;
    border-radius: 15px !important;
    padding: 0 22px !important;
    font-size: 15px !important;

    background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
    box-shadow:
        0 18px 42px rgba(6,182,212,0.28),
        inset 0 1px 0 rgba(255,255,255,0.25) !important;
}

.analysis-report-card {
    max-width: 1100px;
    margin: 0 auto 70px auto !important;

    padding: 38px 42px !important;
    border-radius: 34px !important;

    background:
        radial-gradient(circle at 15% 0%, rgba(207,250,254,0.40), transparent 28%),
        radial-gradient(circle at 90% 8%, rgba(224,242,254,0.48), transparent 26%),
        rgba(255,255,255,0.82) !important;

    border: 1px solid rgba(255,255,255,0.95) !important;

    box-shadow:
        0 30px 90px rgba(15,23,42,0.10),
        inset 0 1px 0 rgba(255,255,255,0.95) !important;

    backdrop-filter: blur(24px);
}

.report-section-title {
    font-size: 22px !important;
    margin-bottom: 26px !important;
}

.report-section-title span {
    width: 10px !important;
    height: 10px !important;
    background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
    box-shadow: 0 0 0 6px rgba(6,182,212,0.10);
}

.analysis-report-card p {
    max-width: 920px;
    font-size: 16.5px !important;
    line-height: 2 !important;
    color: #475569 !important;
}

.analysis-report-card hr {
    margin: 34px 0 !important;
    border-top: 1px solid rgba(148,163,184,0.22) !important;
}

.small-title {
    font-size: 15px !important;
    font-weight: 900 !important;
    letter-spacing: 0.08em !important;
}

.analysis-report-card ul {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px 26px;
}

.analysis-report-card li {
    margin: 0 !important;
    padding: 16px 18px 16px 34px !important;

    border-radius: 18px;
    background: rgba(248,250,252,0.72);
    border: 1px solid rgba(226,232,240,0.75);

    font-size: 15px !important;
    color: #475569 !important;
}

.analysis-report-card li::before {
    left: 18px !important;
    top: 25px !important;
    background: #0891b2 !important;
}

.analysis-report-card strong {
    color: #0f172a !important;
    font-weight: 900 !important;
}
                
.report-actions button {
    height: 44px !important;
    min-height: 44px !important;
    max-height: 44px !important;

    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;

    padding: 0 18px !important;
    margin: 0 !important;

    vertical-align: middle !important;
    line-height: 44px !important;

    position: static !important;
    top: auto !important;
    transform: none !important;
}

.report-actions button * {
    line-height: 1 !important;
    vertical-align: middle !important;
}
                
.report-actions button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}

.btn-icon {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    width: 16px !important;
    height: 16px !important;

    margin-top: -1px !important;

    line-height: 1 !important;
}

/* ===== METRIC CARD FIX ===== */

.metric-box {
    position: relative !important;
    overflow: hidden !important;

    min-height: 250px !important;
    padding: 26px 24px !important;

    border-radius: 30px !important;
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid rgba(255,255,255,0.9) !important;
    backdrop-filter: blur(20px) !important;

    box-shadow:
        0 20px 60px rgba(15,23,42,0.08),
        inset 0 1px 0 rgba(255,255,255,0.95) !important;
}

.metric-box::before {
    content: "" !important;
    position: absolute !important;

    width: 220px !important;
    height: 220px !important;

    border-radius: 999px !important;
    filter: blur(50px) !important;

    top: -70px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
}

.metric-head {
    position: relative !important;
    display: flex !important;
    align-items: center !important;
    gap: 14px !important;
    z-index: 2 !important;
}

.metric-icon {
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;

    border-radius: 16px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    font-size: 24px !important;

    background: linear-gradient(135deg, #06b6d4, #2563eb) !important;
    box-shadow: 0 14px 32px rgba(6,182,212,0.25) !important;
}

.metric-title {
    margin: 0 !important;

    font-size: 16px !important;
    font-weight: 800 !important;

    color: #111827 !important;
    
    line-height: 1.2 !important;
}

.metric-risk {
    display: block !important;

    margin-top: 6px !important;

    font-size: 15px !important;
    font-weight: 700 !important;

    line-height: 1.2 !important;

    color: #d97706 !important;
}

.metric-value {
    position: relative !important;

    width: 145px !important;
    height: 145px !important;

    margin: 28px auto 0 auto !important;

    border-radius: 999px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    font-size: 25px !important;
    font-weight: 900 !important;

    color: #0891b2 !important;

    line-height: 1 !important;
    text-align: center !important;
    white-space: nowrap !important;
    padding: 0 !important;

    background:
        radial-gradient(circle at center,
        rgba(255,255,255,0.96),
        rgba(240,249,255,0.92)) !important;

    border: 9px solid rgba(14,165,233,0.18) !important;

    box-shadow:
        inset 0 0 20px rgba(255,255,255,0.9),
        0 10px 40px rgba(14,165,233,0.16) !important;

    z-index: 2 !important;
}

.metric-value::before {
    content: "" !important;

    position: absolute !important;

    inset: -9px !important;

    border-radius: 999px !important;

    border-top: 10px solid #0ea5e9 !important;
    border-right: 10px solid #0ea5e9 !important;
    border-bottom: 10px solid transparent !important;
    border-left: 10px solid transparent !important;

    transform: rotate(35deg) !important;
}

.metric-cnn::before {
    background: radial-gradient(circle, rgba(14,165,233,0.18), transparent 65%) !important;
}

.metric-lstm::before {
    background: radial-gradient(circle, rgba(217,70,239,0.18), transparent 65%) !important;
}

.metric-final::before {
    background: radial-gradient(circle, rgba(249,115,22,0.20), transparent 65%) !important;
}
                
/* ===== ACTIVE STREAMLIT SIDEBAR MENU ===== */

[data-testid="stSidebar"] {
    display: block !important;

    position: fixed !important;

    top: 0 !important;
    left: 0 !important;

    width: 300px !important;
    min-width: 300px !important;
    height: 100vh !important;

    background: #0f172a !important;

    z-index: 999999 !important;

    border-right: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 36px 24px !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 14px;
    margin-top: 28px;
}

[data-testid="stSidebar"] label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 18px !important;
    padding: 16px 18px !important;
    color: #cbd5e1 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] label:hover {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(34,211,238,0.25) !important;
}

[data-testid="stSidebar"] label:has(input:checked) {
    background: rgba(34,211,238,0.15) !important;
    border-color: rgba(34,211,238,0.35) !important;
}

[data-testid="stSidebar"] label:has(input:checked) p {
    color: #22d3ee !important;
}

[data-testid="stSidebar"] input {
    display: none !important;
}

[data-testid="stSidebar"] p {
    color: #e5e7eb !important;
    font-weight: 700 !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 14px;
    padding-bottom: 30px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.sidebar-footer {
    margin-top: 28px;
    padding: 16px;
    border-radius: 18px;
    background: rgba(255,255,255,0.04);
    color: #4ade80 !important;
    text-align: center;
    font-weight: 800;
}
                
/* ===== STREAMLIT LAYOUT CENTER FIX ===== */

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

.main .block-container,
.block-container {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    padding-top: 1.2rem !important;
    transition: all 0.25s ease !important;
}

/* Sidebar açıkken: içerik kalan alanın içinde ortalı kalsın */
[data-testid="stSidebar"][aria-expanded="true"] ~ [data-testid="stAppViewContainer"] .block-container {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Sidebar kapalıyken: solda boşluk bırakma */
[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    position: fixed !important;
    top: 76px !important;
    left: 16px !important;
    z-index: 9999999 !important;
    background: #0f172a !important;
    border-radius: 999px !important;
    box-shadow: 0 10px 30px rgba(15,23,42,0.25) !important;
}

[data-testid="stSidebar"][aria-expanded="false"] ~ [data-testid="stAppViewContainer"] {
    margin-left: 0 !important;
}

/* Üst bar */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

header[data-testid="stHeader"] * {
    color: #cbd5e1 !important;
}

header[data-testid="stHeader"] button {
    background: rgba(15, 23, 42, 0.72) !important;
    border-radius: 12px !important;
    color: #cbd5e1 !important;
}

header[data-testid="stHeader"] svg {
    fill: #22d3ee !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}  
                
                
.report-actions button {
    min-width: 132px !important;
}

.btn-icon {
    display: inline-block !important;
    width: 18px !important;
    height: 18px !important;
    line-height: 18px !important;
    text-align: center !important;
    font-size: 14px !important;
}

.report-actions .share-icon {
    position: relative !important;
    top: 1px !important;
}

.report-actions button {
    width: 120px !important;
    height: 44px !important;
}

.report-actions {
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
}

.report-actions button {
    width: 132px !important;
    height: 44px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    padding: 0 !important;
    margin: 0 !important;

    border-radius: 14px !important;
    border: 1px solid rgba(226,232,240,0.85) !important;
    background: rgba(255,255,255,0.86) !important;

    color: #0f172a !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    line-height: 1 !important;

    box-shadow:
        0 10px 28px rgba(15,23,42,0.06),
        inset 0 1px 0 rgba(255,255,255,0.95) !important;
}

.report-actions button span {
    display: block !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.report-actions button {
    width: 132px !important;
    height: 44px !important;
    padding: 0 !important;
    margin: 0 !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.report-actions button span {
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}

            
</style>
""", unsafe_allow_html=True)                    

    


