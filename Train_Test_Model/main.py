import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Train_Test_Model/carpricesBMW.csv")
print(df.head())

# Plot to see the relationship Sell Price vs Age
sns.relplot(data=df, y='Sell Price($)', x='Age(yrs)')
plt.show()

# Plot to see the relationship Sell Price vs Mileage
sns.relplot(data=df, y='Sell Price($)', x='Mileage')
plt.show()

# Prepare variables
X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']

# Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #selects random sample, add random_state=10 to keep sample sample

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))