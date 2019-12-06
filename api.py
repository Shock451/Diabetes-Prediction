from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

# {"Pregnancies" : 10, "Glucose" : 101, "BloodPressure" : 76, "SkinThickness" : 48, "Insulin" : 180, "BMI" : 32.9, "DiabetesPedigreeFunction" : 0.171, "Age" : 63}	

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame([json_]))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    lr = joblib.load("C:/Users/HP/Documents/Clean Start/Py/AI Project/model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("C:/Users/HP/Documents/Clean Start/Py/AI Project/model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(debug=True)