# app.py - Final version (complete, ready-to-run)
# Place this file in the same folder as:
#   - pcos_final_pipeline.joblib
#   - feature_columns.json  (or feature_columns.joblib)
#
# Run with:
#   python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# ---------- UI styling (larger font + nicer spacing) ----------
st.set_page_config(page_title="PCOS Risk Predictor", page_icon="🩺", layout="centered")
st.markdown(
    """
    <style>
    html, body, .stApp { font-size: 18px; }
    h1 { font-size: 34px; }
    h2 { font-size: 26px; }
    .stButton>button { height: 50px; font-size: 18px; }
    .metric-value { font-size: 28px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PCOS Risk Predictor — Demo")
st.markdown("**Note:** This is a demo tool — not a medical diagnosis. Consult a clinician for medical decisions.")

# ---------------- Files expected in same folder ----------------
MODEL_PATH = "pcos_final_pipeline.joblib"
FEATURES_JSON = "feature_columns.json"
FEATURES_JOBLIB = "feature_columns.joblib"

@st.cache_resource
def load_model_and_features():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Place pcos_final_pipeline.joblib in this folder.")
        return None, None
    model = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON, "r") as f:
            features = json.load(f)
    elif os.path.exists(FEATURES_JOBLIB):
        features = joblib.load(FEATURES_JOBLIB)
    else:
        try:
            pre = model.named_steps.get("pre") or model.named_steps.get("preprocessor")
            features = list(pre.feature_names_in_)
        except Exception:
            st.error("feature_columns.json / feature_columns.joblib not found and could not infer feature names.")
            return model, None
    return model, features

model, feature_columns = load_model_and_features()
if model is None or feature_columns is None:
    st.stop()

# ---------------- Exact mapping to your dataset column names ----------------
# These strings must match the columns used at training time (including spaces).
COL_MAP = {
    "Age": " Age (yrs)",
    "Weight": "Weight (Kg)",
    "Height": "Height(Cm) ",
    "BMI": "BMI",
    "Blood Group": "Blood Group",
    "Pulse": "Pulse rate(bpm) ",
    "RR": "RR (breaths/min)",
    "Hb": "Hb(g/dl)",
    "Cycle_RI": "Cycle(R/I)",
    "Cycle_len": "Cycle length(days)",
    "Marriage_years": "Marraige Status (Yrs)",
    "Pregnant": "Pregnant(Y/N)",
    "Abortions": "No. of aborptions",
    "FSH": "FSH(mIU/mL)",
    "LH": "LH(mIU/mL)",
    "FSH_LH": "FSH/LH",
    "Hip": "Hip(inch)",
    "Waist": "Waist(inch)",
    "WHR": "Waist:Hip Ratio",
    "TSH": "TSH (mIU/L)",
    "AMH": "AMH(ng/mL)",
    "PRL": "PRL(ng/mL)",
    "VitD3": "Vit D3 (ng/mL)",
    "PRG": "PRG(ng/mL)",
    "RBS": "RBS(mg/dl)",
    "WeightGain": "Weight gain(Y/N)",
    "HairGrowth": "hair growth(Y/N)",
    "SkinDark": "Skin darkening (Y/N)",
    "HairLoss": "Hair loss(Y/N)",
    "Pimples": "Pimples(Y/N)",
    "FastFood": "Fast food (Y/N)",
    "RegExercise": "Reg.Exercise(Y/N)",
    "BPsys": "BP _Systolic (mmHg)",
    "BPdia": "BP _Diastolic (mmHg)",
    "Fol_L": "Follicle No. (L)",
    "Fol_R": "Follicle No. (R)",
    "AvgF_L": "Avg. F size (L) (mm)",
    "AvgF_R": "Avg. F size (R) (mm)",
    "Endometrium": "Endometrium (mm)"
}

# ---------------- Helpers ----------------
def yn_to_int(val):
    if val is None: return np.nan
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ["yes","y","1","true","t"]: return 1
        if v in ["no","n","0","false","f"]: return 0
    if isinstance(val, (int, float)):
        return int(val)
    return np.nan

def safe_float(s):
    if s is None or (isinstance(s, str) and s.strip()==""):
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def build_input_df(user_dict, feature_columns):
    row = {c: np.nan for c in feature_columns}
    # direct logical-to-col mapping
    for logical, colname in COL_MAP.items():
        if logical in user_dict and colname in row:
            row[colname] = user_dict[logical]
    # fallback normalization: strip spaces and lowercase
    norm_map = {c.replace(" ","").lower(): c for c in row.keys()}
    for k, v in list(user_dict.items()):
        key_norm = k.replace(" ","").lower()
        if key_norm in norm_map:
            row[norm_map[key_norm]] = v
    return pd.DataFrame([row], columns=feature_columns)

def risk_level(proba):
    if proba < 0.20:
        return "🟢 Low Risk"
    elif proba < 0.50:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

# ---------------- FORM ----------------
with st.form("pcos_form"):
    st.header("Basic Information (required)")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (yrs)", min_value=10, max_value=70, value=25)
        weight = st.number_input("Weight (Kg)", min_value=20.0, max_value=200.0, value=60.0, format="%.2f")
        height = st.number_input("Height (Cm)", min_value=100.0, max_value=210.0, value=160.0, format="%.2f")
    with c2:
        blood_group = st.selectbox("Blood Group", ["", "A+","A-","B+","B-","O+","O-","AB+","AB-"])
        pulse = st.number_input("Pulse rate (bpm)", min_value=30, max_value=200, value=72)
        cycle_ri = st.selectbox("Cycle (R/I)", ["Regular","Irregular"])
    with c3:
        cycle_len = st.number_input("Cycle length (days)", min_value=0, max_value=120, value=28)
        marriage_years = st.number_input("Marriage Status (Years)", min_value=0, max_value=60, value=0)
        pregnant = st.selectbox("Pregnant (Y/N)", ["No","Yes"])

    st.header("Symptoms (required)")
    s1, s2, s3 = st.columns(3)
    with s1:
        abortions = st.number_input("No. of abortions", min_value=0, max_value=20, value=0)
        hip = st.number_input("Hip (inch)", min_value=10.0, max_value=70.0, value=36.0, format="%.2f")
        waist = st.number_input("Waist (inch)", min_value=10.0, max_value=70.0, value=30.0, format="%.2f")
    with s2:
        hair_growth = st.selectbox("Hair growth (Y/N)", ["No","Yes"])
        skin_dark = st.selectbox("Skin darkening (Y/N)", ["No","Yes"])
        hair_loss = st.selectbox("Hair loss (Y/N)", ["No","Yes"])
    with s3:
        pimples = st.selectbox("Pimples (Y/N)", ["No","Yes"])
        fast_food = st.selectbox("Fast food (Y/N)", ["No","Yes"])
        reg_ex = st.selectbox("Regular Exercise (Y/N)", ["No","Yes"])

    st.markdown("---")
    st.header("Optional Medical Tests (leave blank if not available)")
    o1, o2, o3 = st.columns(3)
    with o1:
        rr = st.text_input("RR (breaths/min)", value="")
        hb = st.text_input("Hb(g/dl)", value="")
        fsh = st.text_input("FSH(mIU/mL)", value="")
        lh = st.text_input("LH(mIU/mL)", value="")
        fsh_lh = st.text_input("FSH/LH", value="")
        tsh = st.text_input("TSH (mIU/L)", value="")
    with o2:
        amh = st.text_input("AMH(ng/mL)", value="")
        prl = st.text_input("PRL(ng/mL)", value="")
        vitd = st.text_input("Vit D3 (ng/mL)", value="")
        prg = st.text_input("PRG(ng/mL)", value="")
        rbs = st.text_input("RBS(mg/dl)", value="")
    with o3:
        bp_sys = st.text_input("BP _Systolic (mmHg)", value="")
        bp_dia = st.text_input("BP _Diastolic (mmHg)", value="")
        fol_l = st.text_input("Follicle No. (L)", value="")
        fol_r = st.text_input("Follicle No. (R)", value="")
        avgf_l = st.text_input("Avg. F size (L) (mm)", value="")
        avgf_r = st.text_input("Avg. F size (R) (mm)", value="")
        endo = st.text_input("Endometrium (mm)", value="")

    consent_save = st.checkbox("Save anonymized input for demo (no name)", value=False)
    submit = st.form_submit_button("Predict", type="primary")

# ---------------- ON SUBMIT ----------------
if submit:
    # Validate required fields
    missing_required = []
    if blood_group == "":
        missing_required.append("Blood Group")
    if missing_required:
        st.error("Please fill required fields: " + ", ".join(missing_required))
    else:
        # compute BMI safely
        try:
            BMI = float(weight) / ((float(height)/100.0)**2) if height>0 else np.nan
        except:
            BMI = np.nan

        # parse optional numeric inputs
        rr_v = safe_float(rr); hb_v = safe_float(hb); fsh_v = safe_float(fsh); lh_v = safe_float(lh)
        fshlh_v = safe_float(fsh_lh); tsh_v = safe_float(tsh); amh_v = safe_float(amh)
        prl_v = safe_float(prl); vitd_v = safe_float(vitd); prg_v = safe_float(prg); rbs_v = safe_float(rbs)
        bp_sys_v = safe_float(bp_sys); bp_dia_v = safe_float(bp_dia)
        fol_l_v = safe_float(fol_l); fol_r_v = safe_float(fol_r)
        avgf_l_v = safe_float(avgf_l); avgf_r_v = safe_float(avgf_r); endo_v = safe_float(endo)

        # WHR
        try:
            whr_v = float(waist) / float(hip) if hip and hip>0 else np.nan
        except:
            whr_v = np.nan

        # ---------- Blood Group handling ----------
        # Quick safe approach: treat textual blood group as missing (imputer will fill)
        # If you know numeric codes used during training, set USE_BG_AS_MISSING=False and fill BG_MAP.
        USE_BG_AS_MISSING = True
        BG_MAP = {
            # Example: "A+": 15, "B+": 13  (only set if you know the numeric codes from training)
        }
        if USE_BG_AS_MISSING:
            bg_val = np.nan
        else:
            bg_val = BG_MAP.get(blood_group, np.nan)

        # ---------- Cycle mapping: convert Regular/Irregular -> numeric (model expects numeric) ----------
        cycle_map = {"Regular": 0, "Irregular": 1}
        cycle_val = cycle_map.get(cycle_ri, np.nan)

        # Pregnant numeric
        pregnant_val = yn_to_int(pregnant)

        # Build user dictionary (logical keys)
        user_dict = {
            "Age": int(age),
            "Weight": float(weight),
            "Height": float(height),
            "BMI": float(BMI) if BMI is not None else np.nan,
            "Blood Group": bg_val,
            "Pulse": int(pulse),
            "RR": rr_v,
            "Hb": hb_v,
            "Cycle_RI": cycle_val,
            "Cycle_len": int(cycle_len),
            "Marriage_years": int(marriage_years),
            "Pregnant": pregnant_val,
            "Abortions": int(abortions),
            "FSH": fsh_v,
            "LH": lh_v,
            "FSH_LH": fshlh_v,
            "Hip": float(hip),
            "Waist": float(waist),
            "WHR": whr_v,
            "TSH": tsh_v,
            "AMH": amh_v,
            "PRL": prl_v,
            "VitD3": vitd_v,
            "PRG": prg_v,
            "RBS": rbs_v,
            "WeightGain": np.nan,
            "HairGrowth": yn_to_int(hair_growth),
            "SkinDark": yn_to_int(skin_dark),
            "HairLoss": yn_to_int(hair_loss),
            "Pimples": yn_to_int(pimples),
            "FastFood": yn_to_int(fast_food),
            "RegExercise": yn_to_int(reg_ex),
            "BPsys": bp_sys_v,
            "BPdia": bp_dia_v,
            "Fol_L": fol_l_v,
            "Fol_R": fol_r_v,
            "AvgF_L": avgf_l_v,
            "AvgF_R": avgf_r_v,
            "Endometrium": endo_v
        }

        # Build dataframe aligned to model features
        X_user = build_input_df(user_dict, feature_columns)

        # Predict using the loaded pipeline
        try:
            proba = model.predict_proba(X_user)[0,1]
            pred = int(proba >= 0.5)
        except Exception as e:
            st.error("Prediction failed. Model and feature columns may not align: " + str(e))
            st.stop()

        # Show results
        st.subheader("Prediction")
        st.metric("Predicted PCOS probability", f"{proba:.3f}")
        st.write("Risk level:", risk_level(proba))
        if pred == 1:
            st.write("🔴 **Predicted: PCOS**")
        else:
            st.write("🟢 **Predicted: No PCOS**")

        # Save anonymized input if consented
        if consent_save:
            log = X_user.copy()
            log["pred_proba"] = proba
            log["pred_label"] = pred
            log["ts"] = datetime.utcnow().isoformat()
            log_path = "pcos_input_logs.csv"
            if os.path.exists(log_path):
                log.to_csv(log_path, mode="a", header=False, index=False)
            else:
                log.to_csv(log_path, index=False)
            st.success(f"Saved anonymized input to {log_path}")

        # Optional SHAP explanation (if tree model & shap installed)
        try:
            import shap
            st.subheader("Model explanation (SHAP)")
            pre = model.named_steps.get("pre") or model.named_steps.get("preprocessor")
            clf = model.named_steps.get("clf")
            X_trans = pre.transform(X_user)
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_trans)
            feat_names = pre.get_feature_names_out()
            vals = np.abs(shap_values).mean(axis=0)
            top_idx = np.argsort(vals)[::-1][:8]
            for i in top_idx:
                st.write(f"- **{feat_names[i]}**: {vals[i]:.4f}")
        except Exception:
            # skip SHAP silently if not available or incompatible
            pass

st.markdown("---")
st.write("© PCOS Predictor (Demo)")
