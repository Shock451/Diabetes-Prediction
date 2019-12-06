import pandas as pd
import numpy as np

# import data
# file = 'diabetes2.csv'
data = pd.read_csv('C:/Users/HP/Documents/Clean Start/Py/AI Project/diabetes2.csv')
# data = pd.read_csv(file)

# select features
# variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

X = data[variables]
y = data.Outcome

# data processing
# categoricals = []

# for col, col_type in data_.dtypes.iteritems():
#      if col_type == 'O':
#           categoricals.append(col)
#      else:
#           data_[col].fillna(0, inplace=True)

# data_ohe = pd.get_dummies(data_, columns=categoricals, dummy_na=True)

# from sklearn.model_selection import train_test_split
# X_train, X_test,  y_train , y_test = train_test_split(X, y, shuffle = True)

# make your model
from sklearn.ensemble import GradientBoostingClassifier
# dependent_variable = 'Outcome'
# X = data_ohe[data_ohe.columns.difference([dependent_variable])]
# y = data_ohe[dependent_variable]
lr = GradientBoostingClassifier()
lr.fit(X, y)

# Save your model
from sklearn.externals import joblib
joblib.dump(lr, 'C:/Users/HP/Documents/Clean Start/Py/AI Project/model.pkl')
print("Model dumped!")

# Load the model that you just savedl
lr = joblib.load('C:/Users/HP/Documents/Clean Start/Py/AI Project/model.pkl')

# Saving the data columns from training
model_columns = list(X.columns)
joblib.dump(model_columns, 'C:/Users/HP/Documents/Clean Start/Py/AI Project/model_columns.pkl')
print("Models columns dumped!")