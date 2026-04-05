import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f1117; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.hero {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
    border: 1px solid #2a2f3e;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { font-size: 2.8rem; color: #f0c040; margin-bottom: 0.5rem; }
.hero p { color: #8892a4; font-size: 1.1rem; font-weight: 300; }

.card {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.card h3 {
    color: #f0c040;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

.result-approved {
    background: linear-gradient(135deg, #0d2b1f, #1a3a2a);
    border: 2px solid #2ecc71;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-rejected {
    background: linear-gradient(135deg, #2b0d0d, #3a1a1a);
    border: 2px solid #e74c3c;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-approved h2 { color: #2ecc71; font-size: 2rem; }
.result-rejected h2 { color: #e74c3c; font-size: 2rem; }
.result-approved p, .result-rejected p { color: #aab; font-size: 1rem; margin-top: 0.5rem; }

.metric-box {
    background: #0f1117;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-box .value { font-size: 1.4rem; font-weight: 700; color: #f0c040; }
.metric-box .label { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.05em; }

.score-bar-bg {
    background: #0f1117;
    border-radius: 10px;
    height: 14px;
    margin-top: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 14px;
    border-radius: 10px;
    transition: width 0.5s ease;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #c8d0de !important;
    font-weight: 400;
}

.stButton > button {
    background: linear-gradient(135deg, #f0c040, #e0a820);
    color: #0f1117;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.3s ease;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(240, 192, 64, 0.3);
}

.tip-box {
    background: #1a1f2e;
    border-left: 3px solid #f0c040;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    color: #8892a4;
    font-size: 0.9rem;
}

.score-section {
    background: #1a1f2e;
    border: 1px solid #2a2f3e;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}
.score-section h3 { color: #f0c040; margin-bottom: 1rem; font-family: 'DM Sans', sans-serif; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists('best_model.pkl'):
        return joblib.load('best_model.pkl')
    return None

model = load_model()

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏦 Loan Approval</h1>
    <p>AI-powered prediction system — know your chances instantly</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model not found! Please ensure `best_model.pkl` is in the same folder.")
    st.stop()

# ── Personal Details ──────────────────────────────────────────
st.markdown('<div class="card"><h3>👤 Personal Details</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
with col2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
with col3:
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4, 5])

col4, col5 = st.columns(2)
with col4:
    loan_type = st.selectbox("Loan Type", ["Personal", "Home", "Car", "Education", "Gold"])
with col5:
    job_years = st.number_input("Years in Current Job", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
st.markdown('</div>', unsafe_allow_html=True)

# ── Financial Details ─────────────────────────────────────────
st.markdown('<div class="card"><h3>💰 Financial Details</h3>', unsafe_allow_html=True)
col6, col7 = st.columns(2)
with col6:
    monthly_income = st.number_input(
        "Monthly Income (₹)", min_value=5000, max_value=1000000,
        value=50000, step=5000
    )
    income = monthly_income * 12
    st.caption(f"Annual Income: ₹{income:,.0f}")
with col7:
    loan_amount = st.number_input(
        "Loan Amount (₹)", min_value=100000, max_value=50000000,
        value=1500000, step=100000
    )

col8, col9 = st.columns(2)
with col8:
    existing_loans = st.number_input(
        "Existing Loan Amount (₹)", min_value=0, max_value=10000000,
        value=0, step=50000, help="Total outstanding loans you already have"
    )
with col9:
    cibil = st.slider("CIBIL Score", min_value=300, max_value=900, value=700, step=1)
st.markdown('</div>', unsafe_allow_html=True)

# ── Computed Features ─────────────────────────────────────────
loan_income_ratio  = loan_amount / income
emi_income_ratio   = (loan_amount / 60) / income
income_per_person  = income / (dependents + 1)
total_loan_burden  = (loan_amount + existing_loans) / income

# ── Live Metrics ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    cibil_status = "Excellent 🟢" if cibil >= 750 else ("Good 🟡" if cibil >= 650 else "Poor 🔴")
    st.markdown(f'<div class="metric-box"><div class="value">{cibil}</div><div class="label">CIBIL Score</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-box"><div class="value">{loan_income_ratio:.1f}x</div><div class="label">Loan/Income</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-box"><div class="value">{emi_income_ratio*100:.1f}%</div><div class="label">EMI/Income</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-box"><div class="value">₹{income_per_person/12:,.0f}</div><div class="label">Income/Person/Mo</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────
if st.button("🔍 Predict Loan Approval"):

    edu_encoded = 1 if education == "Graduate" else 0
    user_input  = np.array([[dependents, edu_encoded, income,
                             loan_amount, cibil,
                             loan_income_ratio, emi_income_ratio, income_per_person]])

    prediction  = model.predict(user_input)[0]
    proba       = model.predict_proba(user_input)[0]
    confidence  = max(proba) * 100

    # ── AI Result ────────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            <h2>✅ Loan Approved!</h2>
            <p>AI Confidence: <strong>{confidence:.1f}%</strong></p>
            <p>Based on your profile, your loan is likely to be approved.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <h2>❌ Loan Rejected</h2>
            <p>AI Confidence: <strong>{confidence:.1f}%</strong></p>
            <p>Based on your profile, your loan may not be approved.</p>
        </div>""", unsafe_allow_html=True)

    # ── Scoring System ────────────────────────────────────────
    score = 0
    score_details = []

    # Age (10)
    if 21 <= age <= 55:
        score += 10; score_details.append(("Age", 10, 10, "✅"))
    elif 56 <= age <= 65:
        score += 5;  score_details.append(("Age", 5, 10, "🟡"))
    else:
        score_details.append(("Age", 0, 10, "❌"))

    # CIBIL (30)
    if cibil >= 750:
        score += 30; score_details.append(("CIBIL Score", 30, 30, "✅"))
    elif cibil >= 700:
        score += 20; score_details.append(("CIBIL Score", 20, 30, "🟡"))
    elif cibil >= 650:
        score += 10; score_details.append(("CIBIL Score", 10, 30, "🟡"))
    else:
        score_details.append(("CIBIL Score", 0, 30, "❌"))

    # Loan Ratio (20)
    if loan_income_ratio < 0.3:
        score += 20; score_details.append(("Loan/Income Ratio", 20, 20, "✅"))
    elif loan_income_ratio < 0.5:
        score += 10; score_details.append(("Loan/Income Ratio", 10, 20, "🟡"))
    else:
        score_details.append(("Loan/Income Ratio", 0, 20, "❌"))

    # Income per person (15)
    if income_per_person > 300000:
        score += 15; score_details.append(("Income/Person", 15, 15, "✅"))
    elif income_per_person > 150000:
        score += 10; score_details.append(("Income/Person", 10, 15, "🟡"))
    else:
        score += 5;  score_details.append(("Income/Person", 5, 15, "🟡"))

    # Job Stability (15)
    if job_years >= 5:
        score += 15; score_details.append(("Job Stability", 15, 15, "✅"))
    elif job_years >= 2:
        score += 10; score_details.append(("Job Stability", 10, 15, "🟡"))
    else:
        score += 5;  score_details.append(("Job Stability", 5, 15, "🟡"))

    # Total Loan Burden (10)
    if total_loan_burden < 0.4:
        score += 10; score_details.append(("Total Loan Burden", 10, 10, "✅"))
    elif total_loan_burden < 0.7:
        score += 5;  score_details.append(("Total Loan Burden", 5, 10, "🟡"))
    else:
        score_details.append(("Total Loan Burden", 0, 10, "❌"))

    # Education (10)
    if edu_encoded == 1:
        score += 10; score_details.append(("Education", 10, 10, "✅"))
    else:
        score += 5;  score_details.append(("Education", 5, 10, "🟡"))

    max_score = 110
    pct = score / max_score * 100
    bar_color = "#2ecc71" if pct >= 65 else ("#f0c040" if pct >= 45 else "#e74c3c")

    st.markdown(f"""
    <div class="score-section">
        <h3>📊 Risk Score: {score}/{max_score}</h3>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct:.0f}%; background:{bar_color};"></div>
        </div>
        <p style="color:#8892a4; font-size:0.85rem; margin-top:0.5rem;">
            {'Strong Profile 💪' if pct >= 65 else ('Moderate Profile 🤔' if pct >= 45 else 'Weak Profile ⚠️')}
        </p>
    </div>""", unsafe_allow_html=True)

    # Score breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(len(score_details))
    for i, (label, got, total, icon) in enumerate(score_details):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="value" style="font-size:1rem;">{icon} {got}/{total}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── Loan Type Suggestion ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    lt = loan_type.lower()
    if lt == "home" and loan_amount < 500000:
        st.markdown('<div class="tip-box">💡 <strong>Loan Type:</strong> Home loan usually for higher amounts 🏠</div>', unsafe_allow_html=True)
    elif lt == "personal" and loan_amount > 1000000:
        st.markdown('<div class="tip-box">💡 <strong>Loan Type:</strong> Consider Home/Secured Loan for large amounts 🔒</div>', unsafe_allow_html=True)
    elif lt == "gold" and cibil > 750:
        st.markdown('<div class="tip-box">💡 <strong>Loan Type:</strong> With your CIBIL score, you qualify for Personal Loan too 💳</div>', unsafe_allow_html=True)

    # ── Improvement Tips ──────────────────────────────────────
    if cibil < 700:
        st.markdown('<div class="tip-box">💡 <strong>Tip:</strong> Pay existing EMIs on time to improve CIBIL above 750.</div>', unsafe_allow_html=True)
    if loan_income_ratio > 0.5:
        st.markdown('<div class="tip-box">💡 <strong>Tip:</strong> Loan amount is high vs income. Consider reducing loan or increasing income.</div>', unsafe_allow_html=True)
    if total_loan_burden > 0.7:
        st.markdown('<div class="tip-box">💡 <strong>Tip:</strong> Total loan burden is very high. Close existing loans before applying.</div>', unsafe_allow_html=True)
    if job_years < 2:
        st.markdown('<div class="tip-box">💡 <strong>Tip:</strong> Job stability is low. Banks prefer 2+ years in current job.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#4a5060; font-size:0.85rem;">
    Built with ❤️ using Random Forest & Streamlit
</div>
""", unsafe_allow_html=True)
