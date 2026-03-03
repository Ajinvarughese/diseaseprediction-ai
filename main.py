from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form
from services import *
from entity import *
from chatAI import *
import json

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
        if(disease == "diabetes"):
            statements = extract_values_from_pdf_diabetes(pdf_bytes)
        else:    
            statements = extract_values_from_pdf(pdf_bytes, disease)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return statements


@app.post("/ai/chat")
async def chat(
    user_input: str = Form(...),
    user: str = Form(...),
    chat_log: str = Form("[]"),
    patient_report: UploadFile | None = File(None),
):
    extracted_patient_report = ""
    try:
        user_data = json.loads(user)
        chat_log_data = json.loads(chat_log)
        print(patient_report)
        if patient_report:
            pdf_bytes = await patient_report.read()
            extracted_patient_report = extract_pdf_of_chat(pdf_bytes)
            print(extracted_patient_report)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return askAI(
        patient_report=extracted_patient_report,
        user_input=user_input,
        chat_log=chat_log_data,
        user=user_data,
    )
