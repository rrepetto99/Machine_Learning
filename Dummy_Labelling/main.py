# We assign dummy variables to condict multifactor regression
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Dummy_Labelling/carprices.csv")

#Plot 
sns.relplot(data=df, x="Age(yrs)", y="Sell Price($)", hue="Car Model")
plt.show()

# Create dummy variables using pandas
dummies = pd.get_dummies(df[['Car Model']], dtype=int)
#print(dummies)

#Merge data set with dummy variables
df_filtered = df.drop(['Car Model'], axis='columns')
df_merged = pd.concat([df_filtered, dummies], axis='columns')

# We drop Audi column to avoid multicolinearity
df_final = df_merged.drop(['Car Model_Audi A5'], axis='columns')

# Now we aaign variables
X = df_final.drop(['Sell Price($)'], axis='columns')
y = df_final[['Sell Price($)']]

# Conduct Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X, y)

# Predict price of a mercedez benz that is 4 yr old with mileage 45000 
price_of_mercedes = model.predict([[45000, 4, 0, 1]]) 
print(f"Price of Mercedes is {np.round(price_of_mercedes, 2)} USD")

# Predict price of a BMW X5 that is 7 yr old with mileage 86000
price_of_BMW = model.predict([[86000, 7, 1, 0]]) 
print(f"Price of BMW is {np.round(price_of_BMW, 2)} USD")

R_squared = model.score(X,y)
print(f"R_squared is {np.round(R_squared, 5)}")

# Second method to use OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])
print(dfle)

# We create variables
X = dfle[['Car Model', 'Mileage', 'Age(yrs)']].values
y = dfle[['Sell Price($)']].values

print(X)

# Now we use OHE to create dummy variables for each of the car
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('Sell Price($)', OneHotEncoder(), [0])], remainder='passthrough')

X = ct.fit_transform(X)
print(X)

# We remove first column
X = X[:, 1:]
print(X)
# Test the model
model.fit(X, y)
price_of_mercedes = model.predict([[0, 1, 45000, 4]]) 
print(f"Price of Mercedes is {np.round(price_of_mercedes, 2)} USD")




