from fastapi import FastAPI
import joblib
import pandas as pd 
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")
MODEL_PATH= "models/churn_model.pkl"
model= joblib.load(MODEL_PATH)

@app.get("/")
def health_check():
    return{"status": "ok",
    "message" :"Churn Prediction API is running"} 


class CustomerData(BaseModel):
    features:dict
    
@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.features])
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}

    
    