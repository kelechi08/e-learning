"""
E-Learning Adoption Predictor (Streamlit)
- Batch or single-row prediction using saved artifacts
- Includes a polished blue theme (sliders, radios, selects, tabs, buttons)
Run:
    streamlit run streamlit_app.py
"""
import os, json, base64
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import joblib

ARTIFACT_DIR = "artifacts"
BEST_MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_model.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler.joblib")
ENCODERS_PATH = os.path.join(ARTIFACT_DIR, "label_encoders.joblib")
FEATURE_LIST_PATH = os.path.join(ARTIFACT_DIR, "feature_list.json")
SHAP_SUMMARY_PNG = os.path.join(ARTIFACT_DIR, "shap_summary.png")
LIME_HTML = os.path.join(ARTIFACT_DIR, "lime_explanation.html")
FI_PNG = os.path.join(ARTIFACT_DIR, "feature_importance.png")

def apply_custom_theme():
    st.markdown("""
    <style>
    .stApp{background:#F7F9FC!important;color:#1B263B!important}
    h1,h2,h3{color:#1565C0!important;font-weight:700!important}
    /* Slider rail + track + thumb + value */
    div[data-testid="stSlider"]>div{background:transparent!important}
    div[data-testid="stSlider"] [class*="rail"]{background:#E0E0E0!important;height:6px!important;border-radius:3px!important}
    div[data-testid="stSlider"] [class*="track"]{background:#1E88E5!important;height:6px!important;border-radius:3px!important}
    div[data-testid="stSlider"] [class*="thumb"]{background:#1565C0!important;border:3px solid #fff!important;width:20px!important;height:20px!important;border-radius:50%!important;box-shadow:0 3px 6px rgba(30,136,229,0.45)!important}
    div[data-testid="stThumbValue"]{background:#1E88E5!important;color:#fff!important;padding:4px 8px!important;border-radius:4px!important;font-weight:600!important;font-size:13px!important}
    /* Force-blue overrides for inline / pseudo elements inside BaseWeb slider */
    [data-baseweb="slider"] [style*="rgb(255, 75, 75)"],
    [data-baseweb="slider"] [style*="#FF4B4B"],
    [data-baseweb="slider"] div[aria-valuenow],
    [data-baseweb="slider"] div[aria-valuenow]::before,
    [data-baseweb="slider"] *[style*="255"]{background-color:#1E88E5!important}
    /* Radios */
    div[role="radiogroup"]{display:flex!important;flex-direction:row!important;gap:40px!important;align-items:center!important}
    div[role="radiogroup"] label{display:flex!important;align-items:center!important;gap:10px!important;margin:0!important;padding:6px 2px!important;cursor:pointer!important}
    div[role="radiogroup"] label div[class*="container"]>div:first-child{width:20px!important;height:20px!important;border:2px solid #90CAF9!important;border-radius:50%!important;background:#fff!important;display:flex!important;align-items:center!important;justify-content:center!important}
    div[role="radiogroup"] input:checked + div > div:first-child{background:#1E88E5!important;border-color:#1E88E5!important}
    div[role="radiogroup"] input:checked + div > div:first-child::after{content:'';width:8px;height:8px;background:#fff;border-radius:50%}
    /* Multi-select tags, selects, inputs, uploader, buttons, tabs, form, sidebar */
    span[data-baseweb="tag"]{background:#E3F2FD!important;color:#1565C0!important;border:1px solid #90CAF9!important;border-radius:16px!important;padding:6px 14px!important;font-size:14px!important;font-weight:500!important}
    div[data-baseweb="select"]>div{border-radius:8px!important;border-color:#D1D5DB!important}
    div[data-baseweb="select"]>div:hover{border-color:#64B5F6!important}
    div[data-baseweb="select"]>div:focus-within{border-color:#1E88E5!important;box-shadow:0 0 0 1px #1E88E5!important}
    input[type="text"],input[type="number"],textarea,.stTextInput input{border:1px solid #1E88E5!important;border-radius:6px!important}
    section[data-testid="stFileUploader"]>div>div{border:2px dashed #90CAF9!important;background:#F8FBFF!important;border-radius:8px!important;padding:1.5rem!important}
    section[data-testid="stFileUploader"]:hover>div>div{border-color:#42A5F5!important}
    button[kind="primary"],button[kind="primaryFormSubmit"]{background:#1E88E5!important;border:none!important;color:#fff!important;border-radius:6px!important;font-weight:600!important;padding:0.6rem 1.25rem!important}
    button[kind="primary"]:hover{background:#1565C0!important}
    button[data-baseweb="tab"]{color:#5B6773!important;font-weight:500!important;padding:0.75rem!important}
    button[data-baseweb="tab"][aria-selected="true"]{color:#1E88E5!important;font-weight:700!important}
    [data-baseweb="tab-highlight"]{background:#1E88E5!important;height:3px!important}
    [data-testid="stForm"]{border:1px solid #E5E7EB!important;background:#fff!important;border-radius:12px!important;padding:28px!important}
    [data-testid="stSidebar"]{background:#F4F7FB!important}
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="E-Learning Adoption Predictor", layout="wide", initial_sidebar_state="expanded", page_icon="🎓")
apply_custom_theme()
st.title("🎓 E-Learning Adoption Predictor")
st.markdown("Predict e-learning adoption using a saved model.")
missing = [p for p in (BEST_MODEL_PATH, SCALER_PATH, ENCODERS_PATH, FEATURE_LIST_PATH) if not os.path.exists(p)]
if missing:
    st.error(f"Missing artifacts: {missing}. Generate them with the model pipeline.")
    st.stop()
try:
    best_model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoders = joblib.load(ENCODERS_PATH)
    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_list = json.load(f)
    st.sidebar.success("✅ Model artifacts loaded")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

def preprocess_apply_streamlit(df_in: pd.DataFrame):
    df = df_in.copy()
    df = df.drop(columns=[c for c in df.columns if 'timestamp' in c.lower() or 'time' in c.lower()], errors='ignore')
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == object:
                mode = df[col].mode()
                df[col].fillna(mode[0] if len(mode) else "Unknown", inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).factorize()[0]
    cols = df.columns.tolist()
    pu_keywords = ['useful','facilitate','improve','enhance','benefit','comprehension']
    pu_cols = [c for c in cols if any(k in c.lower() for k in pu_keywords) and 'willing' not in c.lower()]
    if pu_cols and 'perceived_usefulness_score' not in df.columns:
        df['perceived_usefulness_score'] = df[pu_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    peou_keywords = ['easy','simple','convenient','effort']
    peou_cols = [c for c in cols if any(k in c.lower() for k in peou_keywords)]
    if peou_cols and 'perceived_ease_score' not in df.columns:
        df['perceived_ease_score'] = df[peou_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    bi_keywords = ['willing','intend','want','plan']
    bi_cols = [c for c in cols if any(k in c.lower() for k in bi_keywords)]
    if bi_cols and 'willingness_score' not in df.columns:
        df['willingness_score'] = df[bi_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(df.median())
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0.0
    df = df[feature_list]
    X_scaled = scaler.transform(df)
    return df, X_scaled

def get_table_download_link(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'

st.sidebar.header("🎯 Prediction Mode")
mode = st.sidebar.selectbox("Mode", ["Upload dataset (Batch Prediction)", "Single-row input (Interactive Form)"])

if mode == "Upload dataset (Batch Prediction)":
    st.header("📊 Batch Prediction")
    uploaded = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            st.subheader("Data Preview")
            st.dataframe(df_in.head(), use_container_width=True)
            if st.button("🚀 Run Predictions", type="primary"):
                with st.spinner("Predicting..."):
                    _, X = preprocess_apply_streamlit(df_in)
                    preds = best_model.predict(X)
                    probs = best_model.predict_proba(X)[:, 1] if hasattr(best_model, "predict_proba") else np.zeros(len(preds))
                    result = df_in.copy()
                    result["Predicted_Adoption"] = ["YES (1)" if p == 1 else "NO (0)" for p in preds]
                    result["Probability_Adoption"] = probs
                st.success("✅ Predictions complete")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total", len(result))
                c2.metric("Likely Adopters", int((result["Predicted_Adoption"] == "YES (1)").sum()))
                c3.metric("Non-Adopters", int((result["Predicted_Adoption"] == "NO (0)").sum()))
                st.dataframe(result[['Predicted_Adoption', 'Probability_Adoption']].head(10), use_container_width=True)
                st.markdown(get_table_download_link(result), unsafe_allow_html=True)
                st.markdown("---")
                st.subheader("💡 Interpretability")
                tab_global, tab_lime = st.tabs(["🌍 Global Feature Impact", "🔍 Local Explanation"])
                with tab_global:
                    if os.path.exists(SHAP_SUMMARY_PNG):
                        st.image(SHAP_SUMMARY_PNG, caption="SHAP Summary", use_container_width=True)
                    if os.path.exists(FI_PNG):
                        st.image(FI_PNG, caption="Feature Importance", use_container_width=True)
                with tab_lime:
                    st.info("Pre-calculated LIME (representative instance)")
                    if os.path.exists(LIME_HTML):
                        with open(LIME_HTML, "r", encoding="utf-8") as f:
                            html = f.read()
                        st.components.v1.html(html, height=480, scrolling=True)
        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.header("👤 Single Instance Prediction")
    st.info("Likert scales: 1 (Strongly Disagree) to 5 (Strongly Agree)")

    original_header_list = [
        'Timestamp','Select your Gender','Select your age range','Select your geographical zone','Select your present Academic Status',
        'Select all your past and present Education Level/Certificates (You can choose more than one)',
        'Please select all social Media Platforms for which you have a personal profile or account (You can choose more than one)',
        'Select all the emerging technologies that you have used or participated in for academic purposes (You can choose more than one)',
        'How do you access your social media platforms? (You can choose more than one)',
        'How long have you been using social networking sites?','How often do you visit your Social Network accounts?',
        '"On average, how much time do you spend daily on a social networking site?"','How many of your classmates are your contacts/friends on your social networking sites?',
        'Why do you use social networks? (You can choose more than one)','Incorporating social media platforms and emerging technologies highlighted earlier will enhance my productivity as a Student',
        'Social media platforms and emerging technologies highlighted earlier could help facilitate learning and engage students at educational institutions?',
        'Incorporating social media and emerging technologies as a teaching tool will increase my comprehension and assimilation as a student',
        'Educational Learning via the social media platforms will be easy for me','Educational Learning via emerging technologies highlighted earlier will be easy for me',
        'It will be beneficial for me to become skilful at using social media platforms and emerging technologies for learning',
        'My colleagues think I should use social media and emerging technologies for learning','My family and friends will appreciate my use of social media and emerging technologies for learning',
        'My privacy will be infringed if social media platforms and emerging technologies are proposed for teaching and learning',
        'Internet availability and signal strength will be a problem in using the social media and emerging technologies for learning\xa0 ',
        'Internet data bundles affordability will be a problem in using the social media and emerging technologies for learning\xa0 ',
        'Power availability will be a problem in using the social media and emerging technologies for learning\xa0 ',
        'I have the technical skills to use social media platforms and emerging technologies for learning','I will be willing to use social media platforms and emerging technologies for learning.',
        'I will be willing to devote the required time and energy for my learning activities via social media platforms and emerging technologies for learning.',
        '"If you are not willing to use nor devote time to social media and emerging technologies for learning, State Why."',
        'Are you presently involved in e-learning using any social media platform or emerging technologies earlier highlighted?',
        'Which elearning courses or platforms are you currently engaged in.'
    ]

    with st.form("single_form"):
        data = {}
        st.subheader("1️⃣ Demographics & Academic Status")
        c1, c2, c3 = st.columns(3)
        with c1:
            data["Select your Gender"] = st.selectbox("Gender", ['Male', 'Female', 'Other'])
            data["Select your age range"] = st.selectbox("Age Range", ['15 - 25', '26 - 35', '36 - 45', '46 - 55', '56+'])
        with c2:
            data["Select your geographical zone"] = st.selectbox("Geographical Zone",
                ['North Central (NC)', 'South West (SW)', 'South South (SS)', 'South East (SE)', 'North East (NE)', 'North West (NW)'])
        with c3:
            data["Select your present Academic Status"] = st.selectbox("Academic Status",
                ['Student - Full time', 'Student - Part time or Distance Learning', 'Non Students'])
        data["Select all your past and present Education Level/Certificates (You can choose more than one)"] = st.multiselect(
            "Education Level",
            ['SSCE', 'OND', 'School of Nursing, Midwifery', 'First Degree (BSc, B.Tech, B.Edu, ...)', "Master's Degree", 'PhD'],
            default=['First Degree (BSc, B.Tech, B.Edu, ...)'])

        st.markdown("---")
        st.subheader("2️⃣ Technology Usage & Habits")
        t1, t2 = st.columns(2)
        with t1:
            data["Please select all social Media Platforms for which you have a personal profile or account (You can choose more than one)"] = st.multiselect(
                "Social Media Platforms",
                ['Facebook', 'Twitter', 'Instagram', 'WhatsApp', 'Telegram', 'LinkedIn', 'YouTube', 'Pinterest', 'Tumblr'],
                default=['Facebook', 'WhatsApp'])
            data["Select all the emerging technologies that you have used or participated in for academic purposes (You can choose more than one)"] = st.multiselect(
                "Emerging Technologies",
                ['Zoom', 'Facebook Live', 'Google Classroom', 'MOOCs (Coursera, Udemy, edX)', 'LMS (Moodle, Talent, Docebo)', 'Web-based Elearning Platforms'],
                default=['Google Classroom'])
            data["How do you access your social media platforms? (You can choose more than one)"] = st.multiselect(
                "Access Device(s)", ['Laptop', 'Smatphone', 'Desktop', 'Tablet'], default=['Smatphone'])
        with t2:
            data["How long have you been using social networking sites?"] = st.selectbox("Usage Duration", ['below 1 year', '1 - 2 years', '2 - 5 years', 'above 5 years'], index=3)
            data["How often do you visit your Social Network accounts?"] = st.selectbox("Visit Frequency", ['Daily', 'Weekly', 'Monthly', 'Rarely'])
            data['"On average, how much time do you spend daily on a social networking site?"'] = st.selectbox("Daily Time Spent", ['Less than 1 hour', '1 - 6 hours per day', 'More than 6 hours per day'], index=1)
            data["How many of your classmates are your contacts/friends on your social networking sites?"] = st.selectbox("Classmates as Contacts", ['None', '1 - 20', '20 - 50', 'Almost everyone'], index=2)

        data["Why do you use social networks? (You can choose more than one)"] = st.multiselect(
            "Reasons for Using Social Networks",
            ['To find information', 'To make professional and business contacts', 'To keep in touch with family and friends',
             'To make new friends', 'To get latest information/gist', 'To share videos/ pictures/ music',
             'To share your experience', 'For academic purposes', 'To while away time'],
            default=['For academic purposes', 'To keep in touch with family and friends'])

        st.markdown("---")
        st.subheader("3️⃣ Perceptions and Willingness")
        likert_labels = {
            "Incorporating social media platforms and emerging technologies highlighted earlier will enhance my productivity as a Student": "Enhances Productivity",
            "Social media platforms and emerging technologies highlighted earlier could help facilitate learning and engage students at educational institutions?": "Facilitates Learning",
            "Incorporating social media and emerging technologies as a teaching tool will increase my comprehension and assimilation as a student": "Increases Comprehension",
            "Educational Learning via the social media platforms will be easy for me": "SM Learning Ease",
            "Educational Learning via emerging technologies highlighted earlier will be easy for me": "ET Learning Ease",
            "It will be beneficial for me to become skilful at using social media platforms and emerging technologies for learning": "Perceived Benefit",
            "I will be willing to use social media platforms and emerging technologies for learning.": "Willingness to Use",
            "I will be willing to devote the required time and energy for my learning activities via social media platforms and emerging technologies for learning.": "Willingness to Devote Time",
        }
        cols_likert = st.columns(2)
        for i, (full, short) in enumerate(likert_labels.items()):
            with cols_likert[i % 2]:
                data[full] = st.slider(short, 1, 5, 4, help=full)

        st.markdown("---")
        st.subheader("4️⃣ Barriers and Social Influence")
        barrier_labels = {
            "My colleagues think I should use social media and emerging technologies for learning": "Colleague Influence",
            "My family and friends will appreciate my use of social media and emerging technologies for learning": "Family/Friend Influence",
            "My privacy will be infringed if social media platforms and emerging technologies are proposed for teaching and learning": "Privacy Concern",
            "Internet availability and signal strength will be a problem in using the social media and emerging technologies for learning\xa0 ": "Internet Availability",
            "Internet data bundles affordability will be a problem in using the social media and emerging technologies for learning\xa0 ": "Data Affordability",
            "Power availability will be a problem in using the social media and emerging technologies for learning\xa0 ": "Power Availability",
            "I have the technical skills to use social media platforms and emerging technologies for learning": "Technical Skills",
        }
        cols_barrier = st.columns(2)
        for i, (full, short) in enumerate(barrier_labels.items()):
            with cols_barrier[i % 2]:
                data[full] = st.slider(short, 1, 5, 3, help=full)

        st.markdown("---")
        st.subheader("5️⃣ Current E-Learning Status")
        data["Are you presently involved in e-learning using any social media platform or emerging technologies earlier highlighted?"] = st.radio(
            "Currently involved in E-Learning?", ['Yes', 'No'], index=1, horizontal=True)
        data['"If you are not willing to use nor devote time to social media and emerging technologies for learning, State Why."'] = st.text_input("Reason for Unwillingness (Optional)", "")
        data["Which elearning courses or platforms are you currently engaged in."] = st.text_input("Current E-Learning Courses/Platforms (Optional)", "")

        submitted = st.form_submit_button("🎯 Predict E-Learning Adoption", type="primary")

    if submitted:
        try:
            for k, v in data.items():
                if isinstance(v, list):
                    data[k] = ";".join(v)
            data['Timestamp'] = pd.to_datetime('today').strftime('%Y/%m/%d %I:%M:%S %p GMT+1')
            ordered = {col: [data.get(col, '')] for col in original_header_list}
            df_in = pd.DataFrame(ordered)
            with st.spinner("Processing..."):
                _, X = preprocess_apply_streamlit(df_in)
                pred = best_model.predict(X)[0]
                prob = best_model.predict_proba(X)[0, 1] if hasattr(best_model, "predict_proba") else None
            st.markdown("---")
            st.subheader("🎉 Prediction Result")
            a1, a2 = st.columns([1, 2])
            with a1:
                if int(pred) == 1:
                    st.balloons()
                    st.success("HIGH ADOPTION")
                else:
                    st.warning("LOW ADOPTION")
            with a2:
                st.metric("Adoption Probability", f"{float(prob):.2%}" if prob is not None else "N/A")
            st.markdown("---")
            st.subheader("💡 Interpretability")
            tab_global, tab_lime = st.tabs(["🌍 Global Impact", "🔍 Local Explanation"])
            with tab_global:
                if os.path.exists(SHAP_SUMMARY_PNG):
                    st.image(SHAP_SUMMARY_PNG, use_container_width=True)
                if os.path.exists(FI_PNG):
                    st.image(FI_PNG, use_container_width=True)
            with tab_lime:
                st.info("Pre-calculated LIME (representative instance)")
                if os.path.exists(LIME_HTML):
                    with open(LIME_HTML, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=480, scrolling=True)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("🔬 App developed using Streamlit | Powered by Machine Learning")