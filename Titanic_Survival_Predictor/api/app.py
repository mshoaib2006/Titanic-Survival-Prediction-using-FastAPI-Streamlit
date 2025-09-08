from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Predictor API",
    description="Predicts survival probability for Titanic passengers",
    version="1.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema 
class PassengerInput(BaseModel):
    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: Literal["S", "C", "Q"]
    Title: Literal["Mr", "Miss", "Mrs", "Master", "Other"]
    HadCabin: bool
    FamilySize: int
    IsAlone: bool

# Load model 
MODEL_PATH = "/home/aidev/Data/Shoaib/Titanic_Survival_Predictor/data/model.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f" Model file not found at: {MODEL_PATH}")

model_package = joblib.load(MODEL_PATH)
model = model_package["model"]
feature_names = model_package.get("features", [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
    'Embarked', 'Title', 'HadCabin', 'FamilySize', 'IsAlone'
])

# Preprocess input
def preprocess_input(data: PassengerInput) -> pd.DataFrame:
    row = data.dict()

    # Encode values as during training
    row['Sex'] = 0 if row['Sex'] == "male" else 1
    row['Embarked'] = {'S': 0, 'C': 1, 'Q': 2}[row['Embarked']]
    row['Title'] = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Other': 4}[row['Title']]
    row['HadCabin'] = int(row['HadCabin'])

    df = pd.DataFrame([row], columns=feature_names)
    return df


@app.post("/predict")
async def predict_survival(passenger: PassengerInput):
    try:
        input_df = preprocess_input(passenger)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "status": "success",
            "prediction": int(prediction),
            "probability": round(probability, 4),
            "interpretation": "Survived" if prediction == 1 else "Did not survive",
            "processed_features": input_df.to_dict(orient="records")[0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Health Check
@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "expected_features": feature_names
    }

#Test Prediction (Dummy Input)
@app.post("/test_prediction", include_in_schema=False)
async def test_prediction():
    test_data = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 25.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
        "Title": "Mr",
        "HadCabin": False,
        "FamilySize": 1,
        "IsAlone": True
    }
    return await predict_survival(PassengerInput(**test_data))

# Dynamic URL Generator
@app.get("/generate_test_urls", tags=["Utilities"])
async def generate_urls(request: Request):
    base = str(request.base_url).rstrip("/")
    return {
        "predict_url": f"{base}/predict",
        "test_prediction_url": f"{base}/test_prediction",
        "health_url": f"{base}/health",
        "docs_url": f"{base}/docs",
        "features": feature_names
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
