import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Myopic Regression Predictor", page_icon="📈", layout="wide")
st.title("Myopic Regression Prediction")
st.write("Estimate the probability of 0.75 regression from patient parameters.")

# ----------------------------
# LOAD MODEL FILES
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_top30_model.pkl")
    imputer = joblib.load("top30_imputer.pkl")
    features = joblib.load("top30_features.pkl")
    return model, imputer, features

try:
    model, imputer, features = load_artifacts()
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# ----------------------------
# DEFAULT VALUES
# ----------------------------
default = {
    "ACD_Apex": 3.2,
    "PRK": 1,
    "Ablation_depth": 80.0,
    "Pre_Sphere": -5.0,
    "Pre_Cylinder": -1.0,
    "Pre_MRSE_calc.": -5.5,
    "Axis_F_flat": 90.0,
    "R_Min_mm": 6.8,
    "Num._Ecc._F": 0.5,
    "Rs_B_mm": 6.5,
    "Rm_B_mm": 6.3,
    "R_Min_B_mm": 5.9,
    "Pupil_Pos_Y": 0.0,
    "KI": 1.05,
    "IHA": 10.0,
    "IHD": 0.02,
    "D10mm_Pachy": 540.0,
    "D10mm_Prog": 1.4,
    "Def._Amp._Max_mm": 1.1,
    "A1_Time_ms": 7.2,
    "HC_Time_ms": 16.5,
    "Radius_mm": 7.3,
    "HC_Deflection_Length_mm": 7.0,
    "A2_Deflection_Length_mm": 7.1,
    "Whole_Eye_Movement_Max_mm": 0.25,
    "PachySlope": 8.0,
    "ARTh": 400.0,
    "bIOP": 15.0,
    "CBI": 0.2,
    "TBI": 0.3,
}

# ----------------------------
# INPUT FORM
# ----------------------------
with st.form("prediction_form"):
    st.subheader("Clinical Parameters")
    col1, col2 = st.columns(2)
    with col1:
        PRK = st.selectbox("PRK", [0, 1], index=1 if default["PRK"] == 1 else 0)
        ACD_Apex = st.number_input("ACD Apex", value=float(default["ACD_Apex"]))
        Ablation_depth = st.number_input("Ablation depth", value=float(default["Ablation_depth"]))
    with col2:
        Pre_Sphere = st.number_input("Pre Sphere", value=float(default["Pre_Sphere"]))
        Pre_Cylinder = st.number_input("Pre Cylinder", value=float(default["Pre_Cylinder"]))
        Pre_MRSE_calc = st.number_input("Pre MRSE (calc.)", value=float(default["Pre_MRSE_calc."]))

    st.subheader("Tomography Parameters")
    col1, col2 = st.columns(2)
    with col1:
        Axis_F_flat = st.number_input("Axis F (flat)", value=float(default["Axis_F_flat"]))
        R_Min_mm = st.number_input("R Min (mm)", value=float(default["R_Min_mm"]))
        Num_Ecc_F = st.number_input("Num. Ecc. F", value=float(default["Num._Ecc._F"]))
        Rs_B_mm = st.number_input("Rs B (mm)", value=float(default["Rs_B_mm"]))
        Rm_B_mm = st.number_input("Rm B (mm)", value=float(default["Rm_B_mm"]))
        R_Min_B_mm = st.number_input("R Min B (mm)", value=float(default["R_Min_B_mm"]))
        Pupil_Pos_Y = st.number_input("Pupil Pos Y", value=float(default["Pupil_Pos_Y"]))
    with col2:
        KI = st.number_input("KI", value=float(default["KI"]))
        IHA = st.number_input("IHA", value=float(default["IHA"]))
        IHD = st.number_input("IHD", value=float(default["IHD"]))
        D10mm_Pachy = st.number_input("D10mm Pachy", value=float(default["D10mm_Pachy"]))
        D10mm_Prog = st.number_input("D10mm Prog", value=float(default["D10mm_Prog"]))
        ARTh = st.number_input("ARTh", value=float(default["ARTh"]))

    st.subheader("Biomechanical Parameters")
    col1, col2 = st.columns(2)
    with col1:
        Def_Amp_Max_mm = st.number_input("Def. Amp. Max [mm]", value=float(default["Def._Amp._Max_mm"]))
        A1_Time_ms = st.number_input("A1 Time [ms]", value=float(default["A1_Time_ms"]))
        HC_Time_ms = st.number_input("HC Time [ms]", value=float(default["HC_Time_ms"]))
        Radius_mm = st.number_input("Radius [mm]", value=float(default["Radius_mm"]))
        HC_Deflection_Length_mm = st.number_input(
            "HC Deflection Length [mm]",
            value=float(default["HC_Deflection_Length_mm"])
        )
    with col2:
        A2_Deflection_Length_mm = st.number_input(
            "A2 Deflection Length [mm]",
            value=float(default["A2_Deflection_Length_mm"])
        )
        Whole_Eye_Movement_Max_mm = st.number_input(
            "Whole Eye Movement Max [mm]",
            value=float(default["Whole_Eye_Movement_Max_mm"])
        )
        PachySlope = st.number_input("PachySlope", value=float(default["PachySlope"]))
        bIOP = st.number_input("bIOP", value=float(default["bIOP"]))
        CBI = st.number_input("CBI", value=float(default["CBI"]))
        TBI = st.number_input("TBI", value=float(default["TBI"]))

    submitted = st.form_submit_button("Estimate probability")

# ----------------------------
# GAUGE CHART FUNCTION
# ----------------------------
def make_gauge(percent: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=percent,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Estimated Probability of Regression", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 20], "color": "#2ecc71"},   # green
                    {"range": [20, 40], "color": "#f1c40f"},  # yellow
                    {"range": [40, 60], "color": "#e67e22"},  # orange
                    {"range": [60, 100], "color": "#e74c3c"}, # red
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": percent
                }
            }
        )
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=60, b=20))
    return fig

# ----------------------------
# PREDICTION
# ----------------------------
if submitted:
    try:
        input_data = {
            "ACD_Apex": ACD_Apex,
            "PRK": PRK,
            "Ablation_depth": Ablation_depth,
            "Pre_Sphere": Pre_Sphere,
            "Pre_Cylinder": Pre_Cylinder,
            "Pre_MRSE_calc.": Pre_MRSE_calc,
            "Axis_F_flat": Axis_F_flat,
            "R_Min_mm": R_Min_mm,
            "Num._Ecc._F": Num_Ecc_F,
            "Rs_B_mm": Rs_B_mm,
            "Rm_B_mm": Rm_B_mm,
            "R_Min_B_mm": R_Min_B_mm,
            "Pupil_Pos_Y": Pupil_Pos_Y,
            "KI": KI,
            "IHA": IHA,
            "IHD": IHD,
            "D10mm_Pachy": D10mm_Pachy,
            "D10mm_Prog": D10mm_Prog,
            "Def._Amp._Max_mm": Def_Amp_Max_mm,
            "A1_Time_ms": A1_Time_ms,
            "HC_Time_ms": HC_Time_ms,
            "Radius_mm": Radius_mm,
            "HC_Deflection_Length_mm": HC_Deflection_Length_mm,
            "A2_Deflection_Length_mm": A2_Deflection_Length_mm,
            "Whole_Eye_Movement_Max_mm": Whole_Eye_Movement_Max_mm,
            "PachySlope": PachySlope,
            "ARTh": ARTh,
            "bIOP": bIOP,
            "CBI": CBI,
            "TBI": TBI,
        }

        input_df = pd.DataFrame([input_data])

        missing_cols = [f for f in features if f not in input_df.columns]
        if missing_cols:
            st.error(f"Missing required features: {missing_cols}")
            st.stop()

        input_df = input_df[features]
        input_df_imputed = pd.DataFrame(imputer.transform(input_df), columns=features)

        prob = float(model.predict_proba(input_df_imputed)[:, 1][0])
        percent = prob * 100

        st.subheader("Result")
        c1, c2 = st.columns([2, 1])

        with c1:
            st.plotly_chart(make_gauge(percent), use_container_width=True)

        with c2:
            st.metric("Estimated probability", f"{percent:.1f}%")

            if percent < 20:
                st.success("Low predicted risk")
            elif percent < 40:
                st.info("Mild to moderate predicted risk")
            elif percent < 60:
                st.warning("Intermediate predicted risk")
            else:
                st.error("High predicted risk")

            st.caption("This is a model-estimated probability, not a guarantee of outcome.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")