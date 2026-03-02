from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from services import *
from entity import *
from chatAI import *

app = FastAPI()

@app.post("/ai/diabetes", response_model=PredictResponse)
async def diabetes(inputData : Diabetes):
    prediction, confidence = predict_diabetes(input_data=inputData)
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
        riskClass=RiskClass(risk_class_str),
        confidence=confidence 
    )

@app.post("/ai/heart", response_model=PredictResponse)
async def heart(inputData : Heart):
    
    prediction, confidence = predict_heart(input_data=inputData)
    
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
        riskClass=RiskClass(risk_class_str),
        confidence=confidence
    )

@app.post("/ai/parkinson", response_model=PredictResponse)
async def parkinsons(inputData: Parkinsons):

    prediction, confidence = predict_parkinsons(input_data=inputData)
    
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
        riskClass=RiskClass(risk_class_str),
        confidence=confidence
    )

@app.post("/file/extract")
async def extractPdf(
    disease: Literal["diabetes", "heart", "parkinson"] = Query(...),
    file: UploadFile = File(...)
):
    pdf_bytes = await file.read()

    try:
        statements = extract_values_from_pdf(pdf_bytes, disease)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return statements