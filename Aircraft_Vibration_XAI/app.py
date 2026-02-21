import streamlit as st
import requests
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from lime_analysis import get_lime_explanation,plot_lime
model_pipeline = joblib.load('models/engine_vibration_xai.pkl')
st.set_page_config(page_title="AI Engine Diagnostic", layout="wide")
st.title("Real-Time Aircraft Health Monitor")

col1, col2 = st.columns(2)

with col1:
    vibration = st.slider("Engine Vibration (mm/s)", 0.0, 6.0, 4.5)
    temp = st.slider("Engine Temperature (°C)", 400.0, 1200.0, 600.0)

with col2:
    rpm = st.slider("Engine Rotation Speed (RPM)", 10000.0, 40000.0, 30000.0)
    noise = st.slider("Noise Level (dB)", 40.0, 150.0, 140.0)

if st.button("RUN ENGINE DIAGNOSTICS"):
    payload = {
        "Engine_Rotation_Speed": rpm,
        "Engine_Vibration": vibration,
        "Noise": noise,
        "Engine_Temperature": temp
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        prediction = response.json()["Risk"]
        
        if prediction == 1:
            st.error("CRITICAL RISK DETECTED")
        else:
            st.success("ENGINE NOMINAL")
            
    except Exception as e:
        st.warning("Could not connect to FastAPI. Is the server running?")
    st.divider()
    st.subheader("Why did the AI make this decision?")
    scaler = model_pipeline.named_steps['scaler']
    model = model_pipeline.named_steps['classifier']
    feature_names = ['Engine Rotation Speed', 'Engine Temperature', 'Engine Vibration', 'Noise']
    
    input_df = pd.DataFrame([[rpm, temp, vibration, noise]], columns=feature_names)
    
    input_scaled = scaler.transform(input_df)
    
    input_for_shap = pd.DataFrame(input_scaled, columns=feature_names)
    
    explainer = shap.Explainer(model, input_for_shap)
    shap_values = explainer(input_for_shap)

    fig, ax = plt.subplots(figsize=(10, 4))
    
    exp = shap.Explanation(
        values=shap_values.values[0, :, 1], 
        base_values=shap_values.base_values[0, 1], 
        data=input_for_shap.iloc[0], 
        feature_names=feature_names
    )
    
    shap.plots.waterfall(exp, show=False)
    st.pyplot(fig)

    training_sample = pd.read_csv('data/aircraft_sensor_data.csv').drop(columns=['Risk']).head(100)

    explanation = get_lime_explanation(
        model=model_pipeline.named_steps['classifier'],
        scaler=model_pipeline.named_steps['scaler'],
        input=input_df,
        feature_names=feature_names,
        training_sample=training_sample
    )
    
    fig_lime = plot_lime(explanation)
    st.pyplot(fig_lime)
    
    def generate_pilot_advisory(prediction, explanation):
       """
       Converts LIME mathematical weights into a natural language advisory.
       """
       top_feature, weight = explanation.as_list()[0]
       
       if prediction == 1:
           msg = f"**CRITICAL ADVISORY**: Risk detected primarily due to **{top_feature.split(' <=')[0].split(' >')[0]}**."
           if "Vibration" in top_feature:
               return msg + " Immediate inspection of engine bearings and turbine blades recommended."
           elif "Temperature" in top_feature:
               return msg + " Monitor for thermal runaway. Reduce throttle to stabilize EGT."
           else:
               return msg + " Check secondary systems for sensor desync."
       else:
           return "**SYSTEM NOMINAL**: All engine parameters are within safe operational envelopes."
       
    # Replace the last line with this for a professional look
    advisory_text = generate_pilot_advisory(prediction, explanation)
    
    if prediction == 1:
        st.error(advisory_text)
    else:
        st.success(advisory_text)