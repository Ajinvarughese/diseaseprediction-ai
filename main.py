from fastapi import FastAPI
from services import *
from entity import *
from chatAI import *

app = FastAPI()

@app.post("/ai/diabetes", response_model=PredictResponse)
async def diabetes(inputData : Diabetes):
    prediction = predict_diabetes(input_data=inputData)
    message = aiRecommendation(
        dataset=inputData,
        risk_class=prediction,
        disease="DIABETES"
    )

    risk_class_str = classifyRiskClass(
        prediction=prediction,
        desc=message,
        disease="DIABETES"
    )

    return PredictResponse(
        result=prediction,
        message=message,
        riskClass=RiskClass(risk_class_str)  
    )

@app.post("/ai/heart", response_model=PredictResponse)
async def heart(inputData : Heart):
    
    prediction = predict_heart(input_data=inputData)
    
    message = aiRecommendation(
        dataset=inputData,
        risk_class=prediction,
        disease="HEART"
    )

    risk_class_str = classifyRiskClass(
        prediction=prediction,
        desc=message,
        disease="HEART"
    )

    return PredictResponse(
        result=prediction,
        message=message,
        riskClass=RiskClass(risk_class_str)
    )

@app.post("/ai/parkinson", response_model=PredictResponse)
async def parkinsons(inputData: Parkinsons):

    prediction = predict_parkinsons(input_data=inputData)
    message = aiRecommendation(
        dataset=inputData,
        risk_class=prediction,
        disease="PARKINSONS"
    )

    risk_class_str = classifyRiskClass(
        prediction=prediction,
        desc=message,
        disease="PARKINSONS"
    )

    return PredictResponse(
        result=prediction,
        message=message,
        riskClass=RiskClass(risk_class_str)
    )