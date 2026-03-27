import streamlit as st
import numpy as np
import pickle
import streamlit.components.v1 as components

st.set_page_config(
    layout="centered",
    page_title="Churn Predictor",
    page_icon="📉",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp { background-color: #F7F8FA; }
#MainMenu, footer, header { visibility: hidden; }

.top-header {
    background: #0F172A;
    color: white;
    padding: 18px 28px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.top-header h1 { font-size: 20px; font-weight: 600; margin: 0; letter-spacing: -0.3px; }
.top-header p  { font-size: 13px; color: #94A3B8; margin: 4px 0 0; }

.section-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 14px;
}

.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-pill {
    flex: 1;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px 16px;
}
.metric-pill .val { font-size: 22px; font-weight: 600; color: #0F172A; font-family: 'DM Mono', monospace; }
.metric-pill .lbl { font-size: 12px; color: #64748B; margin-top: 2px; }

.risk-high   { background:#FEF2F2; color:#DC2626; border:1px solid #FECACA; border-radius:6px; padding:6px 14px; font-size:13px; font-weight:500; display:inline-block; }
.risk-medium { background:#FFFBEB; color:#D97706; border:1px solid #FDE68A; border-radius:6px; padding:6px 14px; font-size:13px; font-weight:500; display:inline-block; }
.risk-low    { background:#F0FDF4; color:#16A34A; border:1px solid #BBF7D0; border-radius:6px; padding:6px 14px; font-size:13px; font-weight:500; display:inline-block; }

.reason-row {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 13px;
    color: #334155;
    line-height: 1.5;
}
.reason-row:last-child { border-bottom: none; }
.reason-dot { width:7px; height:7px; border-radius:50%; margin-top:5px; flex-shrink:0; }
.reason-dot.neg { background: #F87171; }
.reason-dot.pos { background: #4ADE80; }

.action-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #1E293B;
    line-height: 1.5;
}
.action-card.urgent { border-left-color: #EF4444; }
.action-card.warn   { border-left-color: #F59E0B; }
.action-card.good   { border-left-color: #10B981; }

label { font-size: 13px !important; color: #374151 !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        return pickle.load(f)

data = load_model()
pipeline = data["pipeline"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
  <h1>Customer Churn Predictor</h1>
  <p>Logistic Regression · Telco dataset · Scikit-learn pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Customer profile</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
with col2:
    monthly_charges = st.number_input("Monthly charges", 0.0, 200.0, 65.0, step=1.0)

contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Feature encoding — matches training exactly ───────────────────────────────
# Training: X = np.column_stack((tenure, monthly, contract_month, contract_year, contract_2year))
contract_month = 1 if contract == "Month-to-month" else 0
contract_year  = 1 if contract == "One year"       else 0
contract_2year = 1 if contract == "Two year"       else 0

input_data = np.array([[
    tenure,
    monthly_charges,
    contract_month,
    contract_year,
    contract_2year,
]])

# ── Predict ───────────────────────────────────────────────────────────────────
predict_clicked = st.button("Run prediction", use_container_width=True, type="primary")

if predict_clicked:
    prob = pipeline.predict_proba(input_data)[0][1]
    pct  = int(prob * 100)

    if prob > 0.7:
        risk_label   = "High risk"
        risk_class   = "risk-high"
        gauge_color  = "#EF4444"
        action_class = "urgent"
    elif prob > 0.4:
        risk_label   = "Medium risk"
        risk_class   = "risk-medium"
        gauge_color  = "#F59E0B"
        action_class = "warn"
    else:
        risk_label   = "Low risk"
        risk_class   = "risk-low"
        gauge_color  = "#10B981"
        action_class = "good"

    # ── Result card ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-pill">
            <div class="val">{pct}%</div>
            <div class="lbl">Churn probability</div>
          </div>
          <div class="metric-pill">
            <div class="val">{tenure}mo</div>
            <div class="lbl">Tenure</div>
          </div>
        </div>
        <span class="{risk_class}">{risk_label}</span>
        """, unsafe_allow_html=True)

    with res_col2:
        gauge_svg = f"""
        <svg width="180" height="110" viewBox="0 0 180 110">
          <path d="M 20 90 A 70 70 0 0 1 160 90"
                fill="none" stroke="#E2E8F0" stroke-width="14" stroke-linecap="round"/>
          <path d="M 20 90 A 70 70 0 0 1 160 90"
                fill="none" stroke="{gauge_color}" stroke-width="14" stroke-linecap="round"
                stroke-dasharray="{int(220 * prob)} 220"/>
          <text x="90" y="82" text-anchor="middle"
                font-family="DM Mono, monospace" font-size="22" font-weight="600"
                fill="#0F172A">{pct}%</text>
          <text x="90" y="100" text-anchor="middle"
                font-family="DM Sans, sans-serif" font-size="10"
                fill="#94A3B8">churn probability</text>
        </svg>
        """
        components.html(gauge_svg, height=115)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Why this prediction ───────────────────────────────────────────────────
    reasons = []

    if tenure < 12:
        reasons.append(("neg", "Short tenure — new customers churn at higher rates"))
    elif tenure > 36:
        reasons.append(("pos", "Long tenure — strong loyalty signal"))
    else:
        reasons.append(("pos", "Moderate tenure — customer is past the early churn window"))

    if monthly_charges > 70:
        reasons.append(("neg", "High monthly charges — elevated price sensitivity risk"))
    elif monthly_charges < 30:
        reasons.append(("pos", "Low charges — lower financial pressure to leave"))
    else:
        reasons.append(("pos", "Moderate charges — within typical retention range"))

    if contract == "Month-to-month":
        reasons.append(("neg", "Month-to-month contract — no lock-in, easiest to cancel"))
    elif contract == "One year":
        reasons.append(("pos", "One-year contract — moderate commitment reduces churn risk"))
    else:
        reasons.append(("pos", "Two-year contract — strongest retention indicator in this dataset"))

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Why this prediction</div>', unsafe_allow_html=True)
    rows_html = ""
    for sentiment, text in reasons:
        rows_html += f'<div class="reason-row"><div class="reason-dot {sentiment}"></div><span>{text}</span></div>'
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recommended actions ───────────────────────────────────────────────────
    actions = []

    if prob > 0.7:
        if contract == "Month-to-month":
            actions.append("Offer a discount to upgrade to a 1-year or 2-year contract")
        if tenure < 6:
            actions.append("Trigger onboarding check-in — assign a customer success contact")
        if monthly_charges > 70:
            actions.append("Offer a bundle or loyalty discount to reduce perceived cost")
        if not actions:
            actions.append("Immediate outreach required — no single dominant risk factor identified")
    elif prob > 0.4:
        actions.append("Enrol in loyalty programme before next billing cycle")
        actions.append("Send personalised retention email with usage summary")
        if monthly_charges > 50:
            actions.append("Flag for pricing review — borderline price-sensitive profile")
    else:
        actions.append("Candidate for premium upsell — stable, low-risk customer")
        actions.append("Offer referral programme access or loyalty reward")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Recommended actions</div>', unsafe_allow_html=True)
    for action in actions:
        st.markdown(f'<div class="action-card {action_class}">{action}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <p style="font-size:11px; color:#94A3B8; text-align:center; margin-top:8px;">
    Features: tenure · monthly charges · contract type (one-hot) &nbsp;·&nbsp;
    ROC-AUC 0.821 (from-scratch NumPy) / 0.838 (sklearn)
    </p>
    """, unsafe_allow_html=True)
