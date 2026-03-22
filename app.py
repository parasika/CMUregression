import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="Myopic Regression Predictor", page_icon="📈")
st.title("Myopic Regression Prediction")
st.write("Enter patient values and click Predict.")

# -----------------------------------
# Load saved model files
# -----------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_top30_model.pkl")
    imputer = joblib.load("top30_imputer.pkl")
    features = joblib.load("top30_features.pkl")
    return model, imputer, features

model, imputer, features = load_artifacts()

st.write("Loaded features:")
st.write(features)

# -----------------------------------
# Default values for each feature
# Change these if you want
# -----------------------------------
default_values = {
    "ACD_Apex": 3.20,
    "PRK": 1,
    "Ablation_depth": 80.0,
    "Pre_Sphere": -5.00,
    "Pre_Cylinder": -1.00,
    "Pre_MRSE_calc.": -5.50,
    "Axis_F_flat": 90.0,
    "R_Min_mm": 6.80,
    "Num._Ecc._F": 0.50,
    "Rs_B_mm": 6.50,
    "Rm_B_mm": 6.30,
    "R_Min_B_mm": 5.90,
    "Pupil_Pos_Y": 0.00,
    "KI": 1.05,
    "IHA": 10.0,
    "IHD": 0.020,
    "D10mm_Pachy": 540.0,
    "D10mm_Prog": 1.40,
    "Def._Amp._Max_mm": 1.10,
    "A1_Time_ms": 7.20,
    "HC_Time_ms": 16.5,
    "Radius_mm": 7.30,
    "HC_Deflection_Length_mm": 7.00,
    "A2_Deflection_Length_mm": 7.10,
    "Whole_Eye_Movement_Max_mm": 0.25,
    "PachySlope": 8.0,
    "ARTh": 400.0,
    "bIOP": 15.0,
    "CBI": 0.20,
    "TBI": 0.30,
}

# -----------------------------------
# Pretty labels for display
# left side = saved feature name
# right side = nicer label for UI
# -----------------------------------
pretty_labels = {
    "ACD_Apex": "ACD Apex",
    "PRK": "PRK (0=No, 1=Yes)",
    "Ablation_depth": "Ablation depth",
    "Pre_Sphere": "Pre Sphere",
    "Pre_Cylinder": "Pre Cylinder",
    "Pre_MRSE_calc.": "Pre MRSE (calc.)",
    "Axis_F_flat": "Axis F (flat)",
    "R_Min_mm": "R Min (mm)",
    "Num._Ecc._F": "Num. Ecc. F",
    "Rs_B_mm": "Rs B (mm)",
    "Rm_B_mm": "Rm B (mm)",
    "R_Min_B_mm": "R Min B (mm)",
    "Pupil_Pos_Y": "Pupil Pos Y",
    "KI": "KI",
    "IHA": "IHA",
    "IHD": "IHD",
    "D10mm_Pachy": "D10mm Pachy",
    "D10mm_Prog": "D10mm Prog",
    "Def._Amp._Max_mm": "Def. Amp. Max [mm]",
    "A1_Time_ms": "A1 Time [ms]",
    "HC_Time_ms": "HC Time [ms]",
    "Radius_mm": "Radius [mm]",
    "HC_Deflection_Length_mm": "HC Deflection Length [mm]",
    "A2_Deflection_Length_mm": "A2 Deflection Length [mm]",
    "Whole_Eye_Movement_Max_mm": "Whole Eye Movement Max [mm]",
    "PachySlope": "PachySlope",
    "ARTh": "ARTh",
    "bIOP": "bIOP",
    "CBI": "CBI",
    "TBI": "TBI",
}

# -----------------------------------
# Input form
# -----------------------------------
with st.form("prediction_form"):
    st.subheader("Patient Parameters")

    input_data = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        label = pretty_labels.get(feature, feature)
        default = default_values.get(feature, 0.0)

        with col1 if i % 2 == 0 else col2:
            if feature == "PRK":
                input_data[feature] = st.selectbox(label, [0, 1], index=1 if default == 1 else 0)
            else:
                input_data[feature] = st.number_input(label, value=float(default), format="%.6f")

    threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.5, 0.01)
    submitted = st.form_submit_button("Predict")

# -----------------------------------
# Prediction
# -----------------------------------
if submitted:
    try:
        input_df = pd.DataFrame([input_data])

        # Ensure exact feature order
        missing = [f for f in features if f not in input_df.columns]
        extra = [c for c in input_df.columns if c not in features]

        if missing:
            st.error(f"Missing features: {missing}")
        elif extra:
            st.error(f"Unexpected features: {extra}")
        else:
            input_df = input_df[features]
            input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

            prob = float(model.predict_proba(input_df_imputed)[:, 1][0])
            pred = int(prob >= threshold)

            st.subheader("Prediction Result")
            st.write(f"**Predicted probability of 0.75 regression:** {prob:.4f}")
            st.write(f"**Predicted class at threshold {threshold:.2f}:** {pred}")

            if prob < 0.20:
                risk_text = "Low predicted risk"
            elif prob < 0.40:
                risk_text = "Mild to moderate predicted risk"
            elif prob < 0.60:
                risk_text = "Intermediate predicted risk"
            else:
                risk_text = "High predicted risk"

            st.info(risk_text)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# -----------------------------------
# Batch prediction from Excel
# -----------------------------------
st.divider()
st.subheader("Batch Prediction from Excel")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_excel(uploaded_file)

        st.write("Uploaded columns:")
        st.write(batch_df.columns.tolist())

        missing_cols = [f for f in features if f not in batch_df.columns]

        if missing_cols:
            st.error(f"Your Excel file is missing these required columns: {missing_cols}")
        else:
            X_batch = batch_df[features].copy()
            X_batch_imputed = pd.DataFrame(imputer.transform(X_batch), columns=features)

            batch_df["Predicted_Probability"] = model.predict_proba(X_batch_imputed)[:, 1]
            batch_df["Predicted_Class"] = (batch_df["Predicted_Probability"] >= 0.5).astype(int)

            st.success("Prediction complete")
            st.dataframe(batch_df)

            output_file = "predictions.xlsx"
            batch_df.to_excel(output_file, index=False)

            with open(output_file, "rb") as f:
                st.download_button(
                    label="Download predictions.xlsx",
                    data=f,
                    file_name="predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Batch prediction failed: {str(e)}")