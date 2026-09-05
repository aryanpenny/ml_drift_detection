from fastAPI import fastapi

app = FastAPI(title="Churn Prediction API")

@app.get("/")
def health_check():
    return{"status": "ok",
    "message" :"Churn Prediction API is running"}