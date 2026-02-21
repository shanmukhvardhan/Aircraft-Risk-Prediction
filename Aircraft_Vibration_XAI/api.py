from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
app = FastAPI(title="Aircraft Diagnostics")
model_pipeline = joblib.load('models/engine_vibration_xai.pkl')
class SensorData(BaseModel):
    Engine_Rotation_Speed:float
    Engine_Vibration:float 
    Noise:float 
    Engine_Temperature:float
@app.post('/predict')
def predict_risk(data:SensorData):
    df = pd.DataFrame([{'Engine Rotation Speed':data.Engine_Rotation_Speed,
                        'Engine Temperature':data.Engine_Temperature,
                        'Engine Vibration':data.Engine_Vibration,
                        'Noise':data.Noise
                        }])
    prediction = model_pipeline.predict(df)
    return {"Risk":int(prediction[0])}