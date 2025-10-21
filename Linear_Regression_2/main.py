# Multifactor regression
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n

# Import data file
df = pd.read_csv("Linear_Regression_2/hiring.csv")
print(df)
 
# We substitute NaN values
df["experience"] = df["experience"].fillna('zero')
median_test_score = df["test_score(out of 10)"].median()
df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(median_test_score)
print(df)

# We specify variables and convert one of the variables
df["experience"] = df["experience"].apply(w2n.word_to_num)
x_variables = df.drop('salary($)', axis='columns')
y_variable = df['salary($)']

print(df)

# We create Linear regression model
model = linear_model.LinearRegression()
model.fit(x_variables, y_variable)

# We predict the salry for some cases
salary_junior = model.predict([[2,9,6]])
salary_senior = model.predict([[12,10,10]])
print(f"Salary for junior is {salary_junior} and for senior {salary_senior}")

# We print coefficients
b = model.intercept_
m = model.coef_
print(f"Intercept is {np.round(b, 2)}\nSlopes are {np.round(m, 2)}")


# Save Model To a File Using Python Pickle
import pickle

with open('model_pickle', 'wb') as file:
    pickle.dump(model, file)
    
# Load saved model
with open('model_pickle', 'rb') as file:
    mp = pickle.load(file)
    
test_mp = mp.predict([[3, 10, 5]])
print(test_mp)

# Save Trained Model Using joblib
# from sklearn.externals import joblib

# joblib.dump(model, 'model_joblib')

# # Load saved model
# mj = joblib.load('model_joblib')