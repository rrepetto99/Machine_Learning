import pandas as pd 
import numpy as np 

df = pd.read_csv("15.lesson/Melbourne_housing_FULL.csv")
#print(df.head())

# We clean data
df_filter = df.drop(['Address', 'Lattitude', 'Longtitude', 'CouncilArea', 'YearBuilt', 'Date'], axis=1)


#Fill NA values
# Step 1 - filling with zeros five columns
df_filter[['Propertycount', 'Regionname', 'Bathroom', 'Distance', 'Bedroom2', 'Car']] = df_filter[['Propertycount', 'Regionname', 'Bathroom', 'Distance', 'Bedroom2', 'Car']].fillna(0)

# Step 2 - filling with mean values two columns
df_filter[['BuildingArea', 'Landsize']] = df_filter[['BuildingArea', 'Landsize']].fillna(df_filter[['BuildingArea', 'Landsize']].mean())

# Step 3 - drop NA from price column
df_final = df_filter.dropna()

#Now we assign dummy variables to certain features
df_with_dummies = pd.get_dummies(df_final, drop_first=True, dtype=int)


X = df_with_dummies.drop(['Price'], axis=1)
y = df_with_dummies['Price']


# Train Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

print(f"Linear regression: {reg.score(X_test,y_test)}\n") 

# L1 Lasso Regression
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=8.917e+14) #max_iter=100, tol=0.1
lasso_reg.fit(X_train, y_train)
print(f"Lasso regression: {lasso_reg.score(X_test, y_test)}\n")

# L2 Ridge Regression
ridge_reg = linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, y_train)
print(f"Ridge regression: {ridge_reg.score(X_test, y_test)}")