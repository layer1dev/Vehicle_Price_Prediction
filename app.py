import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request
app = Flask(__name__)
host = '0.0.0.0'
port = 5000

@app.route('/')
def hello():
    return ""

@app.route("/checker",methods=['POST'])
def post():
    file_name = 'model.pkl' 
    gb = joblib.load(file_name) 

    mileage = request.form.get('mileage')
    fuel_id = request.form.get('fuel')
    transmission_id = request.form.get('transmission')
    accident = request.form.get('accident')
    no_option = request.form.get('no_option')
    diffs = request.form.get('diffs')

    arr = [mileage,fuel,transmission,accident,no_option,diffs]
    
    scaler = joblib.load('scaler.save') 
    tmp = scaler.transform([arr])

    clean_data = pd.DataFrame(tmp)

    clean_data.to_numpy()

    tmp = clean_data
    prediction=gb.predict(tmp)

    output=round(prediction[0],2)

    return str(output)

if __name__ == '__main__':
    app.run(host=host, port=port)

