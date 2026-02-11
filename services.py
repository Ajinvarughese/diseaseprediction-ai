import os
import pickle
from entity import Diabetes, Heart, Parkinsons


    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))


# Diabetes Prediction Page
def predict_diabetes(input_data : Diabetes):

    # code for Prediction
    diab_diagnosis = ''

    user_input = [input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure, input_data.SkinThickness, input_data.Insulin,
                    input_data.BMI, input_data.DiabetesPedigreeFunction, input_data.Age]

    user_input = [float(x) for x in user_input]

    diab_prediction = diabetes_model.predict([user_input])

    if diab_prediction[0] == 1:
        diab_diagnosis = True
    else:
        diab_diagnosis = False

    return diab_diagnosis



# Heart Disease Prediction Page
def predict_heart(input_data: Heart):

    # code for Prediction
    heart_diagnosis = ''

    user_input = [input_data.age, input_data.sex, input_data.cp, input_data.trestbps, input_data.chol, input_data.fbs, input_data.restecg,
                    input_data.thalach, input_data.exang, input_data.oldpeak, input_data.slope, input_data.ca, input_data.thal]

    user_input = [float(x) for x in user_input]

    heart_prediction = heart_disease_model.predict([user_input])

    if heart_prediction[0] == 1:
        heart_diagnosis = True
    else:
        heart_diagnosis = False

    return heart_diagnosis



# Parkinson's Prediction Page
def predict_parkinsons(input_data: Parkinsons):

    # code for Prediction
    parkinsons_diagnosis = ''

    user_input = [input_data.fo, input_data.fhi, input_data.flo, input_data.Jitter_percent, input_data.Jitter_Abs,
                    input_data.RAP, input_data.PPQ, input_data.DDP, input_data.Shimmer, input_data.Shimmer_dB, input_data.APQ3, input_data.APQ5,
                    input_data.APQ, input_data.DDA, input_data.NHR, input_data.HNR, input_data.RPDE, input_data.DFA, 
                    input_data.spread1, input_data.spread2, input_data.D2, input_data.PPE]

    user_input = [float(x) for x in user_input]

    parkinsons_prediction = parkinsons_model.predict([user_input])

    if parkinsons_prediction[0] == 1:
        parkinsons_diagnosis = True
    else:
        parkinsons_diagnosis = False

    return parkinsons_diagnosis
