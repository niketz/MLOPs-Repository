from prediction_service import prediction
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
from src.get_data import get_data
from src.load_data import load_and_save
from src.split_data import split_and_save_data
from src.train_and_evaluate import train_and_evaluate
from src.log_production_model import log_production_model

import uvicorn
import joblib
import pandas as pd
import numpy as np
import os
import yaml

# params_path = "params.yaml"

app = FastAPI()

# def read_params(config_path):
#     with open (config_path) as yaml_file:
#         config = yaml.safe_load(yaml_file)
#     return config


# def load_model():
#     config = read_params(params_path)
#     model_dir_path = config ['webapp_model_dir']
#     #model_dir = config ['model_dir']
#     model = joblib.load(model_dir_path)
#     return model

# model = load_model()

class InputFeatures(BaseModel):
    fixed_acidity: float 
    volatile_acidity: float 
    citric_acid: float 
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

config_path="params.yaml"

#Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello From MLOps Team'}

# @app.post('/train')
# async def train():
#     get_data(config_path)
#     load_and_save(config_path)
#     split_and_save_data(config_path)
#     train_and_evaluate(config_path)
#     log_production_model(config_path)

#Single Request Prediction endpoint
@app.post('/predict_single')
async def single_wine_quality(data:InputFeatures):
    data = data.dict()
    # fixed_acidity = data['fixed_acidity']
    # volatile_acidity = data['volatile_acidity'] 
    # citric_acid = data['citric_acid'] 
    # residual_sugar = data['residual_sugar']
    # chlorides = data['chlorides']
    # free_sulfur_dioxide = data['free_sulfur_dioxide']
    # total_sulfur_dioxide = data['total_sulfur_dioxide']
    # density = data['density']
    # pH = data['pH']
    # sulphates = data['sulphates']
    # alcohol = data['alcohol']

    # data = np.array([list(data.values())])

    # # prediction = model.predict([[fixed_acidity,volatile_acidity,citric_acid,
    # #             residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,
    # #             sulphates,alcohol]])

    # prediction = model.predict(data)

    # print(prediction[0])

    # return {
    #     'prediction': prediction[0]
    # }

    response = prediction.api_response(data)
    return response


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
