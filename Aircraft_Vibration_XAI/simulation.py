import streamlit as st
import time
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from lime_analysis import get_lime_explanation
import os

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    model_path = os.path.join(current_dir, 'models', 'engine_vibration_xai.pkl')
    data_path = os.path.join(current_dir, 'data', 'aircraft_sensor_data.csv')
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()

    model = joblib.load(model_path)
    data = pd.read_csv(data_path).drop(columns=['Risk']).head(100)
    return model, data

model_pipeline, training_sample = load_assets()
PHASES = [
    {"name": "TAXI & CHECKLIST", "duration": 8, "base": [12000, 450, 1.1, 85]},
    {"name": "V1 - TAKEOFF ROLL", "duration": 12, "base": [38000, 950, 3.8, 145]},
    {"name": "CRUISE (AUTOPILOT)", "duration": 20, "base": [30000, 720, 1.6, 110]},
    {"name": "DESCENT & APPROACH", "duration": 12, "base": [18000, 550, 1.3, 95]}
]

if 'flight_log' not in st.session_state:
    st.session_state.flight_log = []

st.set_page_config(page_title="Aircraft_Risk", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #020202; color: #00FF00; } 
    .main { 
        border: 15px solid #1c1c1c; 
        border-radius: 10px; 
        background-color: #0a0a0a;
        padding: 20px;
    }
    h1, h2, h3, p { font-family: 'Consolas', monospace; color: #00FF00 !important; }
    .stButton>button {
        background-color: #222;
        color: #00FF00;
        border: 2px solid #444;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Aircraft_Prediction")

def create_gauge(value, title, min_val, max_val, danger_start):
    """Creates a professional radial gauge for the cockpit."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18, 'color': '#00FF00'}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickcolor': "#00FF00"},
            'bar': {'color': "#00FF00"},
            'steps': [
                {'range': [min_val, danger_start], 'color': "#003300"}, 
                {'range': [danger_start, max_val], 'color': "#550000"} 
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': danger_start}
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#00FF00"}, height=250, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def generate_pilot_advisory(prediction, explanation):
    """Translates LIME weights into a pilot command."""
    top_feature, weight = explanation.as_list()[0]
    if prediction == 1:
        feature_clean = top_feature.split(' <=')[0].split(' >')[0]
        msg = f" **CRITICAL**: Risk detected due to **{feature_clean}**."
        if "Vibration" in top_feature:
            return msg + " Inspect engine bearings/turbine blades immediately."
        elif "Temperature" in top_feature:
            return msg + " Possible Thermal Runaway. Reduce throttle."
        return msg + " Check sensor synchronization."
    return "**SYSTEM NOMINAL**: Engine parameters stable."


with st.sidebar:
    st.header("CENTER CONSOLE")
    start_btn = st.button("START")

dashboard = st.empty()

if start_btn:
    model_path = os.path.join(base_path, 'models', 'engine_vibration_xai.pkl')
    data_path = os.path.join(base_path, 'data', 'aircraft_sensor_data.csv')
    
    
    model_pipeline = joblib.load(model_path)
    training_sample = pd.read_csv(data_path).drop(columns=['Risk']).head(100)
    feature_names = ['Engine Rotation Speed', 'Engine Temperature', 'Engine Vibration', 'Noise']

    for phase in PHASES:
        start_time = time.time()
        while time.time() - start_time < phase["duration"]:
            rpm, temp, vib, noise = phase["base"]
            rpm += np.random.normal(0, 50); temp += np.random.normal(0, 2)
            vib += np.random.normal(0, 0.05); noise += np.random.normal(0, 1)

            sensor_fail_flag = False
            rpm += np.random.normal(0, 50)
            temp += np.random.normal(0, 2)
            vib += np.random.normal(0, 0.05)
            noise += np.random.normal(0, 1)
            
            
            if np.random.random() < 0.10: 
                vib += np.random.uniform(3.0, 4.0)  
                
            if np.random.random() < 0.05:
                failed_sensor = np.random.choice(["VIB", "TEMP", "RPM", "NOISE"])
                if failed_sensor == "VIB": vib = None
                elif failed_sensor == "TEMP": temp = None
                elif failed_sensor == "RPM": rpm = None
                else: noise = None
                sensor_fail_flag = True

            try:
                input_df = pd.DataFrame([[rpm, temp, vib, noise]], columns=feature_names)
                if input_df.isnull().values.any():
                    raise ValueError("Pydantic Integrity Block: Null Telemetry")

                prediction = model_pipeline.predict(input_df)[0]
                explanation = get_lime_explanation(
                    model_pipeline.named_steps['classifier'], 
                    model_pipeline.named_steps['scaler'], 
                    input_df, feature_names, training_sample
                )
                advisory = generate_pilot_advisory(prediction, explanation)
            except Exception as e:
                prediction = -1 
                advisory = "**SENSOR BUS FAILURE**: Loss of valid telemetry. AI Diagnosis Offline."
            
            if prediction == 1 or prediction == -1:
                log_entry = {
                    "Time": time.strftime("%H:%M:%S"),
                    "Phase": phase["name"],
                    "Status": "CRITICAL" if prediction == 1 else "SENSOR FAIL",
                    "Diagnosis": advisory.split(".")[0],
                    "Vib": f"{vib:.2f}" if vib is not None else "NULL",
                    "Temp": f"{temp:.0f}" if temp is not None else "NULL"
                }
                st.session_state.flight_log.insert(0, log_entry)

            with dashboard.container():
                st.subheader(f"📟 MFD MISSION PHASE: {phase['name']}")
                unique_id = f"{phase['name']}_{time.time()}"
                
                g1, g2, g3, g4 = st.columns(4)
                g1.plotly_chart(create_gauge(vib if vib is not None else 0, "Vibration", 0, 6, 4.5), use_container_width=True, key=f"v_{unique_id}")
                g2.plotly_chart(create_gauge(temp if temp is not None else 400, "EGT (°C)", 400, 1200, 950), use_container_width=True, key=f"t_{unique_id}")
                g3.plotly_chart(create_gauge(rpm if rpm is not None else 0, "N1 (RPM)", 0, 45000, 40000), use_container_width=True, key=f"r_{unique_id}")
                g4.plotly_chart(create_gauge(noise if noise is not None else 40, "Noise (dB)", 40, 160, 140), use_container_width=True, key=f"n_{unique_id}")

                st.markdown("###ADVISORY DISPLAY")
                if prediction == -1: st.warning(advisory)
                elif prediction == 1: st.error(advisory)
                else: st.success(advisory)
                
                st.divider()
                st.subheader("FLIGHT DATA RECORDER (BLACK BOX)")
                if st.session_state.flight_log:
                    st.dataframe(pd.DataFrame(st.session_state.flight_log), use_container_width=True, hide_index=True)
                else:
                    st.info("No anomalies detected. Telemetry stable.")


            time.sleep(1)


