import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="LoanIQ — AI Loan Predictor", page_icon="🏦", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "landing"
if "result" not in st.session_state:
    st.session_state.result = None

@st.cache_resource
def load_model():
    if os.path.exists('best_model.pkl'):
        return joblib.load('best_model.pkl')
    return None

model = load_model()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background-color: #0f1117; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
.landing-hero { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border: 1px solid #2a2f3e; border-radius: 20px; padding: 4rem 2rem; text-align: center; margin-bottom: 2rem; }
.landing-hero h1 { font-size: 3.5rem; color: #f0c040; margin-bottom: 0.5rem; }
.landing-hero p { color: #8892a4; font-size: 1.2rem; font-weight: 300; margin-bottom: 2rem; }
.feature-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 2rem 0; }
.feature-item { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 1.5rem; text-align: center; }
.feature-item .icon { font-size: 2rem; margin-bottom: 0.5rem; }
.feature-item h4 { color: #f0c040; margin-bottom: 0.3rem; font-size: 0.95rem; }
.feature-item p { color: #8892a4; font-size: 0.82rem; margin: 0; }
.stat-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin: 1.5rem 0; }
.stat-box { background: #0f1117; border: 1px solid #2a2f3e; border-radius: 10px; padding: 1.2rem; text-align: center; }
.stat-box .val { font-size: 1.8rem; font-weight: 700; color: #f0c040; }
.stat-box .lbl { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.05em; }
.card { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 1.8rem; margin-bottom: 1.5rem; }
.card h3 { color: #f0c040; font-size: 1rem; margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase; font-family: 'DM Sans', sans-serif; font-weight: 500; }
.app-header { background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%); border: 1px solid #2a2f3e; border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 2rem; text-align: center; }
.app-header h2 { color: #f0c040; font-size: 1.8rem; margin: 0; }
.app-header p { color: #8892a4; margin: 0.3rem 0 0 0; font-size: 0.95rem; }
.dash-approved { background: linear-gradient(135deg, #0d2b1f, #1a3a2a); border: 2px solid #2ecc71; border-radius: 20px; padding: 2.5rem; text-align: center; margin-bottom: 1.5rem; }
.dash-rejected { background: linear-gradient(135deg, #2b0d0d, #3a1a1a); border: 2px solid #e74c3c; border-radius: 20px; padding: 2.5rem; text-align: center; margin-bottom: 1.5rem; }
.dash-approved h1 { color: #2ecc71; font-size: 2.5rem; margin-bottom: 0.3rem; }
.dash-rejected h1 { color: #e74c3c; font-size: 2.5rem; margin-bottom: 0.3rem; }
.dash-approved p, .dash-rejected p { color: #aab; margin: 0.3rem 0; }
.metric-box { background: #0f1117; border: 1px solid #2a2f3e; border-radius: 10px; padding: 1rem; text-align: center; }
.metric-box .value { font-size: 1.3rem; font-weight: 700; color: #f0c040; }
.metric-box .label { font-size: 0.75rem; color: #8892a4; text-transform: uppercase; letter-spacing: 0.05em; }
.score-bar-bg { background: #0f1117; border-radius: 10px; height: 16px; margin-top: 8px; overflow: hidden; }
.score-bar-fill { height: 16px; border-radius: 10px; }
.score-section { background: #1a1f2e; border: 1px solid #2a2f3e; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }
.score-section h3 { color: #f0c040; font-family: 'DM Sans', sans-serif; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 1rem; }
.tip-box { background: #1a1f2e; border-left: 3px solid #f0c040; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin-top: 0.8rem; color: #8892a4; font-size: 0.9rem; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stNumberInput"] label { color: #c8d0de !important; font-weight: 400; }
.stButton > button { background: linear-gradient(135deg, #f0c040, #e0a820); color: #0f1117; font-weight: 700; font-size: 1.1rem; border: none; border-radius: 10px; padding: 0.75rem 2rem; width: 100%; cursor: pointer; font-family: 'DM Sans', sans-serif; }
.stButton > button:hover { box-shadow: 0 8px 25px rgba(240, 192, 64, 0.3); }
</style>
""", unsafe_allow_html=True)


def landing_page():
    st.markdown("""
    <div class="landing-hero">
        <h1>🏦 LoanIQ</h1>
        <p>AI-powered loan approval predictor — know your chances in seconds</p>
        <div class="stat-row">
            <div class="stat-box"><div class="val">96.5%</div><div class="lbl">Model Accuracy</div></div>
            <div class="stat-box"><div class="val">8</div><div class="lbl">Features Analyzed</div></div>
            <div class="stat-box"><div class="val">4269</div><div class="lbl">Records Trained</div></div>
        </div>
    </div>
    <div class="feature-grid">
        <div class="feature-item"><div class="icon">🤖</div><h4>AI Prediction</h4><p>Random Forest with 98.9% ROC-AUC score</p></div>
        <div class="feature-item"><div class="icon">📊</div><h4>Risk Scoring</h4><p>7-factor scoring system out of 110 points</p></div>
        <div class="feature-item"><div class="icon">💡</div><h4>Smart Tips</h4><p>Personalized advice to improve your profile</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Check My Loan Eligibility"):
        st.session_state.page = "app"
        st.rerun()
    st.markdown('<div style="text-align:center; color:#4a5060; font-size:0.85rem; margin-top:2rem;">Built with ❤️ using Random Forest & Streamlit</div>', unsafe_allow_html=True)


def app_page():
    if st.button("← Back to Home"):
        st.session_state.page = "landing"
        st.session_state.result = None
        st.rerun()

    st.markdown('<div class="app-header"><h2>🏦 Loan Approval Predictor</h2><p>Fill in your details to get instant AI prediction</p></div>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model not found! Please ensure best_model.pkl is in the same folder.")
        st.stop()

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

    st.markdown('<div class="card"><h3>💰 Financial Details</h3>', unsafe_allow_html=True)
    col6, col7 = st.columns(2)
    with col6:
        monthly_income = st.number_input("Monthly Income (₹)", min_value=5000, max_value=1000000, value=50000, step=5000)
        income = monthly_income * 12
        st.caption(f"Annual Income: ₹{income:,.0f}")
    with col7:
        loan_amount = st.number_input("Loan Amount (₹)", min_value=100000, max_value=50000000, value=1500000, step=100000)
    col8, col9 = st.columns(2)
    with col8:
        existing_loans = st.number_input("Existing Loan Amount (₹)", min_value=0, max_value=10000000, value=0, step=50000)
    with col9:
        cibil = st.slider("CIBIL Score", min_value=300, max_value=900, value=700, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    effective_loan    = loan_amount + existing_loans
    loan_income_ratio = effective_loan / income
    emi_income_ratio  = (effective_loan / 60) / income
    income_per_person = income / (dependents + 1)
    total_loan_burden = effective_loan / income

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="value">{cibil}</div><div class="label">CIBIL Score</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="value">{loan_income_ratio:.1f}x</div><div class="label">Loan/Income</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="value">{emi_income_ratio*100:.1f}%</div><div class="label">EMI/Income</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="value">₹{income_per_person/12:,.0f}</div><div class="label">Income/Person/Mo</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Predict Loan Approval"):
        edu_encoded = 1 if education == "Graduate" else 0
        user_input  = np.array([[dependents, edu_encoded, income, effective_loan, cibil, loan_income_ratio, emi_income_ratio, income_per_person]])
        prediction  = model.predict(user_input)[0]
        proba       = model.predict_proba(user_input)[0]
        st.session_state.result = {
            "prediction": int(prediction), "confidence": float(max(proba) * 100),
            "age": age, "education": education, "dependents": dependents,
            "loan_type": loan_type, "job_years": job_years, "income": income,
            "loan_amount": loan_amount, "existing_loans": existing_loans, "cibil": cibil,
            "loan_income_ratio": loan_income_ratio, "emi_income_ratio": emi_income_ratio,
            "income_per_person": income_per_person, "total_loan_burden": total_loan_burden,
            "edu_encoded": edu_encoded
        }
        st.session_state.page = "dashboard"
        st.rerun()


def dashboard_page():
    r = st.session_state.result
    col_back, col_new = st.columns(2)
    with col_back:
        if st.button("← Back to Home"):
            st.session_state.page = "landing"; st.session_state.result = None; st.rerun()
    with col_new:
        if st.button("🔄 Try Again"):
            st.session_state.page = "app"; st.session_state.result = None; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if r["prediction"] == 1:
        st.markdown(f'<div class="dash-approved"><h1>✅ Loan Approved!</h1><p style="font-size:1.1rem;">AI Confidence: <strong>{r["confidence"]:.1f}%</strong></p><p>Based on your profile, your loan is likely to be approved.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="dash-rejected"><h1>❌ Loan Rejected</h1><p style="font-size:1.1rem;">AI Confidence: <strong>{r["confidence"]:.1f}%</strong></p><p>Based on your profile, your loan may not be approved.</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="value">{r["cibil"]}</div><div class="label">CIBIL Score</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="value">{r["loan_income_ratio"]:.1f}x</div><div class="label">Loan/Income</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="value">{r["total_loan_burden"]:.1f}x</div><div class="label">Total Burden</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="value">₹{r["income_per_person"]/12:,.0f}</div><div class="label">Income/Person/Mo</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    score = 0
    score_details = []
    if 21 <= r["age"] <= 55:   score += 10; score_details.append(("Age", 10, 10, "✅"))
    elif 56 <= r["age"] <= 65: score += 5;  score_details.append(("Age", 5, 10, "🟡"))
    else:                                    score_details.append(("Age", 0, 10, "❌"))

    if r["cibil"] >= 750:   score += 30; score_details.append(("CIBIL", 30, 30, "✅"))
    elif r["cibil"] >= 700: score += 20; score_details.append(("CIBIL", 20, 30, "🟡"))
    elif r["cibil"] >= 650: score += 10; score_details.append(("CIBIL", 10, 30, "🟡"))
    else:                                 score_details.append(("CIBIL", 0, 30, "❌"))

    if r["loan_income_ratio"] < 0.3:   score += 20; score_details.append(("Loan Ratio", 20, 20, "✅"))
    elif r["loan_income_ratio"] < 0.5: score += 10; score_details.append(("Loan Ratio", 10, 20, "🟡"))
    else:                                            score_details.append(("Loan Ratio", 0, 20, "❌"))

    if r["income_per_person"] > 300000:   score += 15; score_details.append(("Inc/Person", 15, 15, "✅"))
    elif r["income_per_person"] > 150000: score += 10; score_details.append(("Inc/Person", 10, 15, "🟡"))
    else:                                 score += 5;  score_details.append(("Inc/Person", 5, 15, "🟡"))

    if r["job_years"] >= 5:   score += 15; score_details.append(("Job Stability", 15, 15, "✅"))
    elif r["job_years"] >= 2: score += 10; score_details.append(("Job Stability", 10, 15, "🟡"))
    else:                     score += 5;  score_details.append(("Job Stability", 5, 15, "🟡"))

    if r["total_loan_burden"] < 0.5:   score += 10; score_details.append(("Loan Burden", 10, 10, "✅"))
    elif r["total_loan_burden"] < 1.0: score += 5;  score_details.append(("Loan Burden", 5, 10, "🟡"))
    else:                                            score_details.append(("Loan Burden", 0, 10, "❌"))

    if r["edu_encoded"] == 1: score += 10; score_details.append(("Education", 10, 10, "✅"))
    else:                     score += 5;  score_details.append(("Education", 5, 10, "🟡"))

    pct = score / 110 * 100
    bar_color = "#2ecc71" if pct >= 65 else ("#f0c040" if pct >= 45 else "#e74c3c")
    profile_label = "Strong Profile 💪" if pct >= 65 else ("Moderate Profile 🤔" if pct >= 45 else "Weak Profile ⚠️")

    st.markdown(f"""
    <div class="score-section">
        <h3>📊 Risk Score: {score}/110 — {profile_label}</h3>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct:.0f}%; background:{bar_color};"></div></div>
    </div>""", unsafe_allow_html=True)

    cols = st.columns(len(score_details))
    for i, (label, got, total, icon) in enumerate(score_details):
        with cols[i]:
            st.markdown(f'<div class="metric-box"><div class="value" style="font-size:0.95rem;">{icon} {got}/{total}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    tips = []
    if r["cibil"] < 700:              tips.append("💡 Pay existing EMIs on time to improve CIBIL above 750.")
    if r["loan_income_ratio"] > 0.5:  tips.append("💡 Loan is high vs income. Consider reducing loan amount.")
    if r["total_loan_burden"] > 1.0:  tips.append("💡 Total debt exceeds annual income. Close existing loans first.")
    if r["job_years"] < 2:            tips.append("💡 Banks prefer 2+ years in current job for stability.")
    lt = r["loan_type"].lower()
    if lt == "home" and r["loan_amount"] < 500000:    tips.append("💡 Home loans are usually for higher amounts 🏠")
    elif lt == "personal" and r["loan_amount"] > 1000000: tips.append("💡 Consider a secured loan for large amounts 🔒")
    elif lt == "gold" and r["cibil"] > 750:           tips.append("💡 With your CIBIL, you qualify for Personal Loan too 💳")

    if tips:
        st.markdown("<br>", unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

    st.markdown('<div style="text-align:center; color:#4a5060; font-size:0.85rem; margin-top:2rem;">Built with ❤️ using Random Forest & Streamlit</div>', unsafe_allow_html=True)


if st.session_state.page == "landing":
    landing_page()
elif st.session_state.page == "app":
    app_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
