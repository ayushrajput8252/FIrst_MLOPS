from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()
class model_input(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    skinThickness: int
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

#loading save mode
dia_model  = pickle.load(open('model.pkl', 'rb'))

@app.post('/diabetes_prediction')
def diabetes_prediction(input_parameters : model_input):
    #json is depreseaed
    input_data = input_parameters.model_dump_json()
    input_dict = json.loads(input_data)

    Pregnancies = input_dict['Pregnancies']
    Glucose = input_dict['Glucose']
    BloodPressure = input_dict['BloodPressure']
    skinThickness = input_dict['skinThickness']
    Insulin = input_dict['Insulin']
    BMI = input_dict['BMI']
    DiabetesPedigreeFunction = input_dict['DiabetesPedigreeFunction']
    Age = input_dict['Age']

    input_list = [Pregnancies, Glucose, BloodPressure, skinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    input_reshape = [input_list]
    prediction = dia_model.predict(input_reshape)
    if prediction[0] == 1:
        return {'Result': 'The person is diabetic'}
    else : return {'Result': 'The person is not diabetic'}
