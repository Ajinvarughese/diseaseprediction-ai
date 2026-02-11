from fastapi import FastAPI
from services import *
from entity import *
from chatAI import *

app = FastAPI()



@app.post("/diabetes", response_model=PredictResponse)
async def diabetes(inputData : Diabetes):
    prediction = predict_diabetes(input_data=inputData)
    message = aiRecommendation(dataset=inputData, risk_class=prediction, disease="DIABETES")
    return PredictResponse(
        result=prediction,
        message=message
    )
@app.post("/heart", response_model=PredictResponse)
async def heart(inputData : Heart):
    prediction = predict_heart(input_data=inputData)
    message = aiRecommendation(dataset=inputData, risk_class=prediction, disease="HEART")
    return PredictResponse(
        result=prediction,
        message=message
    )


@app.post("/parkinsons", response_model=PredictResponse)
async def parkinsons(inputData : Parkinsons):
    prediction = predict_parkinsons(input_data=inputData)
    message = aiRecommendation(dataset=inputData, risk_class=prediction, disease="PARKINSONS")
    return PredictResponse(
        result=prediction,
        message=message
    )