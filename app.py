import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import base64
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------- FILE PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "artifacts", "models", "encoders.pkl")
ANIME_PATH = os.path.join(BASE_DIR, "assets", "anime.png.png")
# ---------------- IMAGE HELPER ----------------
def get_base64_image(ANIME_PATH):
    if not os.path.exists(ANIME_PATH):
        return None
    try:
        with open(ANIME_PATH, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None


anime_base64 = get_base64_image(ANIME_PATH)

# ---------------- LOAD MODEL / SCALER / ENCODERS ----------------
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(ENCODER_PATH, "rb") as file:
        encoders = pickle.load(file)

except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"❌ Error loading model/scaler/encoders: {e}")
    st.stop()

# ---------------- EXTRACT ENCODER MAPS ----------------
try:
    experience_map = encoders["experience_level"]
    employment_map = encoders["employment_type"]
    company_size_map = encoders["company_size"]
    job_title_map = encoders["job_title"]
    residence_map = encoders["employee_residence"]
    location_map = encoders["company_location"]
except KeyError as e:
    st.error(f"❌ Missing encoder key in encoders.pkl: {e}")
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* ===== APP BACKGROUND ===== */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg, #f8fafc, #eef2ff, #ffffff);
}

/* ===== MAIN CONTAINER ===== */
.block-container{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* ===== HERO CARD ===== */
.hero-card{
    background: linear-gradient(135deg, #ffffff, #eef2ff);
    padding: 30px;
    border-radius: 28px;
    text-align: center;
    color: #111827;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
}

.hero-title{
    font-size: 52px;
    font-weight: 900;
    margin-bottom: 8px;
    color: #111827;
}

.hero-subtitle{
    font-size: 18px;
    color: #4b5563;
    font-weight: 500;
}

/* ===== SIDEBAR CARD ===== */
.sidebar-card{
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 18px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    margin-bottom: 16px;
}

/* ===== INPUT CARD ===== */
.card{
    background: rgba(255,255,255,0.97);
    padding: 28px;
    border-radius: 24px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* ===== SECTION TITLE ===== */
.section-title{
    font-size: 26px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 14px;
}

/* ===== BUTTON ===== */
.stButton > button{
    width: 100%;
    height: 58px;
    border-radius: 18px;
    border: none;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    color: white;
    box-shadow: 0 10px 25px rgba(79,70,229,0.25);
    transition: 0.3s ease;
}

.stButton > button:hover{
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(79,70,229,0.35);
}

/* ===== RESULT SECTION ===== */
.result-wrapper{
    position: relative;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 28px;
    padding: 35px;
    overflow: hidden;
    min-height: 360px;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 14px 40px rgba(79,70,229,0.28);
}

.result-content{
    position: relative;
    z-index: 2;
    color: white;
    width: 55%;
}

.result-title{
    font-size: 20px;
    font-weight: 700;
    opacity: 0.95;
}

.result-salary{
    font-size: 54px;
    font-weight: 900;
    margin-top: 12px;
}

.result-desc{
    font-size: 16px;
    margin-top: 12px;
    opacity: 0.92;
    line-height: 1.7;
}

.result-badge{
    display: inline-block;
    margin-top: 18px;
    background: rgba(255,255,255,0.18);
    padding: 10px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 700;
    backdrop-filter: blur(6px);
}

.result-anime{
    position: absolute;
    right: 20px;
    bottom: 0;
    height: 340px;
    max-width: 40%;
    object-fit: contain;
    z-index: 1;
    opacity: 1;
    filter: drop-shadow(0px 12px 22px rgba(0,0,0,0.25));
}

/* ===== KPI CARDS ===== */
[data-testid="metric-container"]{
    background: #ffffff;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}

[data-testid="metric-container"] label{
    color:#6b7280 !important;
    font-weight:700;
}

[data-testid="metric-container"] [data-testid="stMetricValue"]{
    color:#111827 !important;
    font-weight:900;
}

/* ===== INSIGHT CARDS ===== */
.insight-card{
    background: linear-gradient(135deg, #ffffff, #f9fafb);
    border: 1px solid #e5e7eb;
    border-left: 6px solid #6366f1;
    padding: 20px;
    border-radius: 18px;
    color: #111827;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

/* ===== SUMMARY CARD ===== */
.summary-card{
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    border: 1px solid #e5e7eb;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.06);
    color: #111827;
    line-height: 1.9;
    font-size: 17px;
    margin-top: 10px;
}

/* ===== FOOTER ===== */
.footer-card{
    margin-top: 30px;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    color: #374151;
    box-shadow: 0 8px 20px rgba(0,0,0,0.06);
}

/* ===== TYPING EFFECT ===== */
.typing {
    overflow: hidden;
    border-right: .15em solid #6366f1;
    white-space: nowrap;
    animation:
        typing 3s steps(40,end),
        blink .75s step-end infinite;
    font-size: 22px;
    color: #1f2937;
    font-weight: bold;
    margin-top: 10px;
}

@keyframes typing {
    from {width:0}
    to {width:100%}
}

@keyframes blink {
    from,to {border-color:transparent}
    50% {border-color:#6366f1}
}

label, .stSelectbox label, .stNumberInput label{
    font-weight: 700 !important;
    color: #1f2937 !important;
}

/* ===== MOBILE ===== */
@media (max-width: 900px){
    .result-content{
        width: 100%;
    }
    .result-anime{
        right: -20px;
        height: 240px;
        opacity: 0.20;
        max-width: 60%;
    }
    .result-salary{
        font-size: 42px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <h2 style="margin:0;">🤖 AI Salary Predictor</h2>
        <p style="margin-top:8px;">Smart ML Salary Estimator</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("""
📊 Predict Data Science Salaries

💰 Machine Learning Powered

📈 Premium Salary Dashboard

🎯 Smart Insights + Confidence Score
""")
    st.markdown("---")
    st.success("✅ Model + Encoders Loaded Successfully")

    if anime_base64:
        st.success("🖼️ Anime image loaded from assets folder")
    else:
        st.warning("⚠️ anime.png not found in assets folder")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero-card">
    <div class="hero-title">💰 AI Salary Predictor</div>
    <div class="hero-subtitle">
        Predict Data Science Salaries Using Machine Learning with a Premium Interactive Dashboard
    </div>
</div>
""", unsafe_allow_html=True)

st.info("📊 Fill the employee profile below and click Predict Salary.")

left, right = st.columns([2, 1])

with left:
    st.markdown("""
    <div class='typing'>
        🤖 AI Assistant Ready To Predict Salaries...
    </div>
    """, unsafe_allow_html=True)

with right:

    st.success("🤖 Anime Salary Assistant Online")
    st.write("")   # 👈 empty line (forces next line)

    st.image(r"C:\Users\javam\Desktop\vishu_full__stack\ds_salaries\assets\anime.png.png", width=500 )


# ---------------- INPUT FORM ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📝 Employee Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    work_year = st.number_input("📅 Work Year", min_value=2020, max_value=2035, value=2024)

    experience_text = st.selectbox("🎓 Experience Level", list(experience_map.keys()))
    experience_level = experience_map[experience_text]

    employment_text = st.selectbox("💼 Employment Type", list(employment_map.keys()))
    employment_type = employment_map[employment_text]

    job_title_text = st.selectbox("👨‍💻 Job Title", list(job_title_map.keys()))
    job_title = job_title_map[job_title_text]

with col2:
    residence_text = st.selectbox("🌍 Employee Residence", list(residence_map.keys()))
    employee_residence = residence_map[residence_text]

    remote_ratio = st.slider("🏠 Remote Ratio", min_value=0, max_value=100, value=100, step=25)

    company_location_text = st.selectbox("📍 Company Location", list(location_map.keys()))
    company_location = location_map[company_location_text]

    company_size_text = st.selectbox("🏢 Company Size", list(company_size_map.keys()))
    company_size = company_size_map[company_size_text]

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICT BUTTON ----------------
if st.button("🚀 Predict Salary"):
    st.snow()

    try:
        # Feature order must match training order
        input_data = pd.DataFrame({
            "work_year": [work_year],
            "experience_level": [experience_level],
            "employment_type": [employment_type],
            "job_title": [job_title],
            "employee_residence": [employee_residence],
            "remote_ratio": [remote_ratio],
            "company_location": [company_location],
            "company_size": [company_size]
        })

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Loading animation
        status = st.empty()
        status.info("🤖 AI Assistant analyzing employee profile...")

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        # Predict salary
        prediction = model.predict(input_scaled)
        salary = float(prediction[0])

        status.success("✅ Analysis Completed")

        # Animated counter
        counter = st.empty()
        for i in np.linspace(0, salary, 80):
            counter.metric("💰 Estimated Salary", f"${i:,.0f}")
            time.sleep(0.01)
        counter.metric("💰 Estimated Salary", f"${salary:,.0f}")

        # Derived metrics
        confidence = min(95, max(70, int((salary / 300000) * 100)))

        if salary < 50000:
            category = "🟢 Entry Level"
        elif salary < 120000:
            category = "🟡 Mid Level"
        elif salary < 250000:
            category = "🟠 Senior Level"
        else:
            category = "🔴 Executive Level"

        # ---------------- RESULT SECTION WITH ANIME ----------------
        if anime_base64:
            st.markdown(
                f"""
                <div class="result-wrapper">
                    <div class="result-content">
                        <div class="result-title">💵 Predicted Salary</div>
                        <div class="result-salary">${salary:,.2f}</div>
                        <div class="result-desc">
                            AI-powered salary estimation based on employee profile.<br>
                            Your anime salary assistant has prepared a premium salary result for you ✨
                        </div>
                        <div class="result-badge">
                            {category} • Confidence {confidence}%
                        </div>
                    </div>
                    <img class="result-anime" src="data:image/png;base64,{anime_base64}" alt="anime">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-wrapper">
                    <div class="result-content" style="width:100%;">
                        <div class="result-title">💵 Predicted Salary</div>
                        <div class="result-salary">${salary:,.2f}</div>
                        <div class="result-desc">
                            AI-powered salary estimation based on employee profile.<br>
                            Anime image not found in assets folder.
                        </div>
                        <div class="result-badge">
                            {category} • Confidence {confidence}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.success("✅ Salary Prediction Completed Successfully")

        # ---------------- KPI DASHBOARD ----------------
        st.markdown("## 📊 Salary Dashboard")
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.metric("💰 Predicted Salary", f"${salary:,.0f}")
        with k2:
            st.metric("🏠 Remote Ratio", f"{remote_ratio}%")
        with k3:
            st.metric("📅 Work Year", work_year)
        with k4:
            st.metric("🎯 Confidence", f"{confidence}%")

        # ---------------- AI INSIGHTS ----------------
        st.markdown("## 🤖 AI Insights")

        if remote_ratio == 100:
            st.markdown("""
            <div class="insight-card">
                <h4>🏠 Remote Work Insight</h4>
                <p>Fully remote roles often remain highly attractive in data science, especially when paired with strong technical experience and in-demand job functions.</p>
            </div>
            """, unsafe_allow_html=True)

        if experience_text in ["Senior", "Executive"]:
            st.markdown("""
            <div class="insight-card">
                <h4>🎓 Experience Impact</h4>
                <p>Senior and executive profiles usually command higher salaries due to domain expertise, leadership scope, and stronger business impact.</p>
            </div>
            """, unsafe_allow_html=True)

        if salary > 150000:
            st.markdown("""
            <div class="insight-card">
                <h4>🚀 High Salary Signal</h4>
                <p>This profile falls into a high-earning range, which may reflect premium market demand, advanced experience, or a specialized technical role.</p>
            </div>
            """, unsafe_allow_html=True)

        if salary < 80000:
            st.markdown("""
            <div class="insight-card">
                <h4>📈 Growth Opportunity</h4>
                <p>This looks like an early-career to mid-level range with strong room for growth as experience, project impact, and specialization increase.</p>
            </div>
            """, unsafe_allow_html=True)

        if company_size_text == "Large":
            st.markdown("""
            <div class="insight-card">
                <h4>🏢 Company Size Insight</h4>
                <p>Larger companies can sometimes offer stronger salary bands, especially for specialized or leadership roles, though this varies by region and role.</p>
            </div>
            """, unsafe_allow_html=True)

        # ---------------- CONFIDENCE ----------------
        st.markdown("## 🎯 Prediction Confidence")
        st.metric("AI Confidence Score", f"{confidence}%")
        st.progress(confidence)

        # ---------------- COMPARISON CHART ----------------
        st.markdown("## 📈 Salary Comparison")
        comparison = pd.DataFrame({
            "Category": ["Average DS Salary", "Predicted Salary"],
            "Salary": [120000, salary]
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison["Category"], comparison["Salary"])
        ax.set_ylabel("Salary ($)")
        ax.set_title("Predicted vs Average Salary")
        st.pyplot(fig)
        plt.close(fig)

        # ---------------- CONFIDENCE PIE ----------------
        st.markdown("## 🥧 Salary Confidence Breakdown")
        fig, ax = plt.subplots()
        ax.pie(
            [confidence, 100 - confidence],
            labels=["Confidence", "Uncertainty"],
            autopct="%1.1f%%"
        )
        ax.axis("equal")
        st.pyplot(fig)
        plt.close(fig)

        # ---------------- SALARY STRENGTH ----------------
        st.markdown("## 🚀 Salary Strength")
        meter = min(100, int((salary / 300000) * 100))
        st.progress(meter)

        if meter >= 80:
            st.success("🔥 Exceptional Salary Range")
        elif meter >= 60:
            st.info("🚀 Strong Salary Range")
        elif meter >= 40:
            st.warning("📈 Average Salary Range")
        else:
            st.error("📉 Lower Salary Range")

        # ---------------- SUMMARY ----------------
        st.markdown("## 📋 Prediction Summary")
        st.markdown(
            f"""
            <div class="summary-card">
                <b>📅 Work Year:</b> {work_year} <br>
                <b>🎓 Experience Level:</b> {experience_text} <br>
                <b>💼 Employment Type:</b> {employment_text} <br>
                <b>👨‍💻 Job Title:</b> {job_title_text} <br>
                <b>🌍 Employee Residence:</b> {residence_text} <br>
                <b>🏠 Remote Ratio:</b> {remote_ratio}% <br>
                <b>📍 Company Location:</b> {company_location_text} <br>
                <b>🏢 Company Size:</b> {company_size_text} <br>
                <b>💰 Estimated Salary:</b> ${salary:,.2f} <br>
                <b>🏆 Salary Category:</b> {category} <br>
                <b>🎯 Confidence:</b> {confidence}%
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# ---------------- DEBUG PATHS ----------------
with st.expander("🔍 Debug Paths"):
    st.write("BASE_DIR:", BASE_DIR)
    st.write("MODEL_PATH exists:", os.path.exists(MODEL_PATH))
    st.write("SCALER_PATH exists:", os.path.exists(SCALER_PATH))
    st.write("ENCODER_PATH exists:", os.path.exists(ENCODER_PATH))
    st.write("ANIME_PATH exists:", os.path.exists(ANIME_PATH))
    st.write("MODEL_PATH:", MODEL_PATH)
    st.write("SCALER_PATH:", SCALER_PATH)
    st.write("ENCODER_PATH:", ENCODER_PATH)
    st.write("ANIME_PATH:", ANIME_PATH)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer-card">
    🚀 Built with Streamlit | Machine Learning | AI Analytics
</div>
""", unsafe_allow_html=True)
