import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


#We open the data file
df = pd.read_csv("canada_per_capita_income.csv")
print(df.head())

# We can plot as scatter
x_variable = df['year']
y_variable = df['per capita income (US$)']
plt.xlabel("year")
plt.ylabel("per capita income (US$)")
plt.scatter(x_variable,y_variable)
plt.show()

# We construct Linear Regression model
reg = linear_model.LinearRegression()
reg.fit(x_variable.values.reshape(-1, 1), y_variable)

# We predict income for 2020
income_2020 = reg.predict([[2020]])
print(f"Predicted income for 2020 is {np.round(income_2020, 2)}")


#We print coefficientsb = reg.coef_
b = reg.intercept_
m = reg.coef_
print(f"Intercept is {np.round(b, 2)}\nSlope is {np.round(m, 2)}")
