import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Myopic Regression Predictor", page_icon="📈")
st.title("Myopic Regression Prediction")
st.write("Predict probability of 0.75 regression")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_top30_model.pkl")
    imputer = joblib.load("top30_imputer.pkl")
    features = joblib.load("top30_features.pkl")
    return model, imputer, features

model, imputer, features = load_model()

# ----------------------------
# DEFAULT VALUES
# ----------------------------
default = {
    "ACD_Apex": 3.2,
    "PRK": 1,
    "Ablation_depth": 80,
    "Pre_Sphere": -5,
    "Pre_Cylinder": -1,
    "Pre_MRSE_calc.": -5.5,
    "Axis_F_flat": 90,
    "R_Min_mm": 6.8,
    "Num._Ecc._F": 0.5,
    "Rs_B_mm": 6.5,
    "Rm_B_mm": 6.3,
    "R_Min_B_mm": 5.9,
    "Pupil_Pos_Y": 0,
    "KI": 1.05,
    "IHA": 10,
    "IHD": 0.02,
    "D10mm_Pachy": 540,
    "D10mm_Prog": 1.4,
    "Def._Amp._Max_mm": 1.1,
    "A1_Time_ms": 7.2,
    "HC_Time_ms": 16.5,
    "Radius_mm": 7.3,
    "HC_Deflection_Length_mm": 7.0,
    "A2_Deflection_Length_mm": 7.1,
    "Whole_Eye_Movement_Max_mm": 0.25,
    "PachySlope": 8,
    "ARTh": 400,
    "bIOP": 15,
    "CBI": 0.2,
    "TBI": 0.3
}

# ----------------------------
# GROUPED INPUT UI
# ----------------------------
with st.form("input_form"):

    st.subheader("👁 Clinical")
    col1, col2 = st.columns(2)
    with col1:
        PRK = st.selectbox("PRK", [0, 1], index=1)
        Pre_Sphere = st.number_input("Pre Sphere", value=default["Pre_Sphere"])
        Pre_Cylinder = st.number_input("Pre Cylinder", value=default["Pre_Cylinder"])
    with col2:
        Pre_MRSE = st.number_input("Pre MRSE (calc.)", value=default["Pre_MRSE_calc."])
        Ablation_depth = st.number_input("Ablation depth", value=default["Ablation_depth"])
        ACD_Apex = st.number_input("ACD Apex", value=default["ACD_Apex"])

    st.subheader("🧿 Tomography")
    col1, col2 = st.columns(2)
    with col1:
        Axis = st.number_input("Axis F (flat)", value=default["Axis_F_flat"])
        R_Min = st.number_input("R Min (mm)", value=default["R_Min_mm"])
        Num_Ecc = st.number_input("Num Ecc F", value=default["Num._Ecc._F"])
        Rs_B = st.number_input("Rs B (mm)", value=default["Rs_B_mm"])
        Rm_B = st.number_input("Rm B (mm)", value=default["Rm_B_mm"])
        R_Min_B = st.number_input("R Min B (mm)", value=default["R_Min_B_mm"])
    with col2:
        Pupil = st.number_input("Pupil Pos Y", value=default["Pupil_Pos_Y"])
        KI = st.number_input("KI", value=default["KI"])
        IHA = st.number_input("IHA", value=default["IHA"])
        IHD = st.number_input("IHD", value=default["IHD"])
        D10_Pachy = st.number_input("D10mm Pachy", value=default["D10mm_Pachy"])
        D10_Prog = st.number_input("D10mm Prog", value=default["D10mm_Prog"])

    st.subheader("⚙️ Biomechanics")
    col1, col2 = st.columns(2)
    with col1:
        DefAmp = st.number_input("Def Amp Max", value=default["Def._Amp._Max_mm"])
        A1_Time = st.number_input("A1 Time", value=default["A1_Time_ms"])
        HC_Time = st.number_input("HC Time", value=default["HC_Time_ms"])
        Radius = st.number_input("Radius", value=default["Radius_mm"])
        HC_Def = st.number_input("HC Deflection Length", value=default["HC_Deflection_Length_mm"])
    with col2:
        A2_Def = st.number_input("A2 Deflection Length", value=default["A2_Deflection_Length_mm"])
        WEM = st.number_input("Whole Eye Movement Max", value=default["Whole_Eye_Movement_Max_mm"])
        PachySlope = st.number_input("PachySlope", value=default["PachySlope"])
        ARTh = st.number_input("ARTh", value=default["ARTh"])
        bIOP = st.number_input("bIOP", value=default["bIOP"])
        CBI = st.number_input("CBI", value=default["CBI"])
        TBI = st.number_input("TBI", value=default["TBI"])

    threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.4)

    submit = st.form_submit_button("Predict")

# ----------------------------
# PREDICTION
# ----------------------------
if submit:
    try:
        data = {
            "ACD_Apex": ACD_Apex,
            "PRK": PRK,
            "Ablation_depth": Ablation_depth,
            "Pre_Sphere": Pre_Sphere,
            "Pre_Cylinder": Pre_Cylinder,
            "Pre_MRSE_calc.": Pre_MRSE,
            "Axis_F_flat": Axis,
            "R_Min_mm": R_Min,
            "Num._Ecc._F": Num_Ecc,
            "Rs_B_mm": Rs_B,
            "Rm_B_mm": Rm_B,
            "R_Min_B_mm": R_Min_B,
            "Pupil_Pos_Y": Pupil,
            "KI": KI,
            "IHA": IHA,
            "IHD": IHD,
            "D10mm_Pachy": D10_Pachy,
            "D10mm_Prog": D10_Prog,
            "Def._Amp._Max_mm": DefAmp,
            "A1_Time_ms": A1_Time,
            "HC_Time_ms": HC_Time,
            "Radius_mm": Radius,
            "HC_Deflection_Length_mm": HC_Def,
            "A2_Deflection_Length_mm": A2_Def,
            "Whole_Eye_Movement_Max_mm": WEM,
            "PachySlope": PachySlope,
            "ARTh": ARTh,
            "bIOP": bIOP,
            "CBI": CBI,
            "TBI": TBI
        }

        df = pd.DataFrame([data])[features]

        df_imputed = pd.DataFrame(imputer.transform(df), columns=features)

        prob = model.predict_proba(df_imputed)[:, 1][0]
        pred = int(prob >= threshold)

        st.subheader("📊 Result")
        st.write(f"**Probability:** {prob:.4f}")
        st.write(f"**Prediction:** {pred}")

        if prob < 0.2:
            st.success("Low risk")
        elif prob < 0.4:
            st.info("Moderate risk")
        elif prob < 0.6:
            st.warning("Intermediate risk")
        else:
            st.error("High risk")

    except Exception as e:
        st.error(f"Prediction failed: {e}")