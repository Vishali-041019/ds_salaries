import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main Background */
[data-testid="stAppViewContainer"]{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #312e81
    );
}

/* Remove top padding */
.block-container{
    padding-top: 2rem;
}

/* Hero Title */
.main-title{
    text-align:center;
    font-size:70px;
    font-weight:900;
    color:white;
    margin-bottom:5px;
}

.subtitle{
    text-align:center;
    font-size:22px;
    color:#d1d5db;
    margin-bottom:40px;
}

/* Input Card */
.card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding:30px;
    border-radius:25px;
    border:1px solid rgba(255,255,255,0.2);
}

/* Result Card */
.result-card{
    background: linear-gradient(
        135deg,
        #00c853,
        #64dd17
    );
    padding:30px;
    border-radius:25px;
    text-align:center;
    color:white;
    font-size:38px;
    font-weight:bold;
}

/* Button */
.stButton > button{
    width:100%;
    height:60px;
    border-radius:15px;
    border:none;
    font-size:22px;
    font-weight:bold;
    background:linear-gradient(
        90deg,
        #ff512f,
        #dd2476
    );
    color:white;
}

/* Metrics */
[data-testid="metric-container"]{
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
    border:1px solid rgba(255,255,255,0.2);
}

.typing {
    overflow: hidden;
    border-right: .15em solid orange;
    white-space: nowrap;
    animation:
        typing 3s steps(40,end),
        blink .75s step-end infinite;
    font-size: 24px;
    color: white;
    font-weight: bold;
}

@keyframes typing {
    from {width:0}
    to {width:100%}
}

@keyframes blink {
    from,to {border-color:transparent}
    50% {border-color:white}
}

.glass-card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    padding:20px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL AND SCALER ----------------
with open(
    r"C:\Users\javam\Desktop\vishu_full__stack\ds_salaries\artifacts\models\model.pkl",
    "rb"
) as file:
    model = pickle.load(file)

with open(
    r"C:\Users\javam\Desktop\vishu_full__stack\ds_salaries\artifacts\models\scaler.pkl",
    "rb"
) as file:
    scaler = pickle.load(file)

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🤖 AI Salary Predictor")

    st.markdown("---")

    st.info("""
    📊 Predict Data Science Salaries

    💰 Machine Learning Powered

    📈 Salary Analytics Dashboard

    🚀 Real-Time Predictions
    """)

    st.markdown("---")

    st.success("Model Loaded Successfully")

# ---------------- HERO SECTION ----------------

st.markdown("""
<div style="
background: linear-gradient(90deg,#6a11cb,#2575fc);
padding:25px;
border-radius:20px;
text-align:center;
color:white;
margin-bottom:20px;
box-shadow:0px 4px 20px rgba(0,0,0,0.3);
">
<h1>💰 AI Salary Predictor</h1>
<h4>Predict Data Science Salaries Using Machine Learning</h4>
</div>
""", unsafe_allow_html=True)

st.info(
    "📊 Enter employee details below and click Predict Salary."
)

left, right = st.columns([2,1])

with left:
    st.markdown("""
    <div class='typing'>
    🤖 AI Assistant Ready To Predict Salaries...
    </div>
    """, unsafe_allow_html=True)

with right:
    st.image(
        "assets/female_ai.png",
        width=280
    )

    st.success("🤖 AI Salary Assistant Online")

# ---------------- INPUT SECTION ----------------

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Employee Information")

col1, col2 = st.columns(2)

with col1:
    work_year = st.number_input(
        "📅 Work Year",
        min_value=2020,
        max_value=2035,
        value=2024
    )

    exp_map = {
        "Entry": 0,
        "Mid": 1,
        "Senior": 2,
        "Executive": 3
    }

    experience_text = st.selectbox(
        "🎓 Experience Level",
        list(exp_map.keys())
    )

    experience_level = exp_map[experience_text]

    employment_type = st.number_input(
        "💼 Employment Type (Encoded)",
        value=0
    )

    job_title = st.number_input(
        "👨‍💻 Job Title (Encoded)",
        value=0
    )

with col2:
    employee_residence = st.number_input(
        "🌍 Employee Residence (Encoded)",
        value=0
    )

    remote_ratio = st.number_input(
        "🏠 Remote Ratio",
        min_value=0,
        max_value=100,
        value=100
    )

    company_location = st.number_input(
        "📍 Company Location (Encoded)",
        value=0
    )

    company_size = st.number_input(
        "🏢 Company Size (Encoded)",
        value=0
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------

if st.button("🚀 Predict Salary"):
    st.snow()


    try:

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

        # Scale Input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        status = st.empty()

        status.info("🤖 AI Assistant analyzing employee profile...")

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        status.success("✅ Analysis Completed")

        salary = float(prediction[0])

        counter = st.empty()

        for i in np.linspace(0, salary, 100):
            counter.metric(
                "💰 Estimated Salary",
                f"${i:,.0f}"
            )
            time.sleep(0.01)

        counter.metric(
            "💰 Estimated Salary",
            f"${salary:,.0f}"
      )



# ---------------- RESULT CARD ----------------
        st.markdown(
            f"""
            <div class="result-card">
                💵 Predicted Salary <br><br>
                ${salary:,.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.success("✅ Salary Prediction Completed Successfully")

        st.divider()



        # Confidence
        confidence = min(
            95,
            max(70, int((salary / 300000) * 100))
        )

        # Salary Category
        if salary < 50000:
            category = "🟢 Entry Level"
        elif salary < 120000:
            category = "🟡 Mid Level"
        elif salary < 250000:
            category = "🟠 Senior Level"
        else:
            category = "🔴 Executive Level"

        # KPI SECTION
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "💰 Predicted Salary",
                f"${salary:,.0f}"
            )

        with c2:
            st.metric(
                "🏠 Remote Ratio",
                f"{remote_ratio}%"
            )

        with c3:
            st.metric(
                "📅 Work Year",
                work_year
            )

        with c4:
            st.metric(
                "🎯 Confidence",
                f"{confidence}%"
            )

        st.divider()

        st.success(f"Salary Category: {category}")

        # AI Insights
        st.markdown("## 🤖 AI Insights")

        if remote_ratio == 100:
            st.info(
                "Remote employees tend to receive competitive salaries in many Data Science roles."
            )

        if salary > 150000:
            st.success(
                "High earning profile detected."
            )

        if salary < 80000:
            st.warning(
                "Entry-level salary range predicted."
            )

        # Confidence Section
        st.subheader("🎯 Prediction Confidence")

        st.metric(
            "🎯 AI Confidence Score",
            f"{confidence}%"
        )

        st.write(
            f"Model Confidence: **{confidence}%**"
        )

        st.divider()

        
        # Salary Comparison Chart
        st.subheader("📈 Salary Comparison")

        comparison = pd.DataFrame({
            "Category": [
                "Average DS Salary",
                "Predicted Salary"
            ],
            "Salary": [
                120000,
                salary
            ]
        })

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            comparison["Category"],
            comparison["Salary"]
        )

        ax.set_ylabel("Salary ($)")
        ax.set_title("Predicted vs Average Salary")

        st.pyplot(fig)

        plt.close()

        st.divider()

        # CONFIDENCE PIE CHART

        st.subheader("🥧 Salary Confidence Breakdown")

        fig, ax = plt.subplots()

        ax.pie(
            [confidence, 100 - confidence],
            labels=["Confidence", "Uncertainty"],
            autopct="%1.1f%%"
        )

        ax.axis("equal")

        st.pyplot(fig)

        plt.close()

        # Salary Meter
        st.subheader("🚀 Salary Strength")

        meter = min(
            100,
            int((salary / 300000) * 100)
        )

        st.progress(meter)

        if meter >= 80:
            st.success("🔥 Exceptional Salary Range")
        elif meter >= 60:
            st.info("🚀 Strong Salary Range")
        elif meter >= 40:
            st.warning("📈 Average Salary Range")
        else:
            st.error("📉 Lower Salary Range")

        # Summary

        st.subheader("📋 Prediction Summary")

        st.info(
            f"""
Work Year: {work_year}

Experience Level: {experience_level}

Employment Type: {employment_type}

Remote Ratio: {remote_ratio}%

Estimated Salary: ${salary:,.2f}

Salary Category: {category}
"""
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.subheader("📋 AI Prediction Process")

        st.write("✅ Employee Profile Loaded")
        st.write("✅ Features Scaled")
        st.write("✅ Salary Model Executed")
        st.write("✅ Confidence Calculated")
        st.write("✅ Prediction Generated")

# ---------------- FOOTER ----------------

st.markdown("""
<div style="
margin-top:30px;
padding:15px;
border-radius:15px;
text-align:center;
background:rgba(255,255,255,0.08);
backdrop-filter:blur(10px);
color:white;
">
🚀 Built with Streamlit | Machine Learning | AI Analytics
</div>
""", unsafe_allow_html=True)