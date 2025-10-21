import pandas as pd 
import numpy as np
import seaborn as sns 
from matplotlib import pyplot as plt 

# Load Dataset
df = pd.read_csv("Decision_Tree/titanic.csv")


# Define Variables
inputs = df[["Pclass", "Sex", "Age", "Fare"]]
target = df[["Survived"]]
#print(inputs.head())

#Encode a dummy to sex column - another way with LabelEncoder
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()

inputs["Sex_n"] = le_sex.fit_transform(inputs["Sex"])
inputs_final = inputs.drop(['Sex'], axis='columns')
print(inputs_final.head())
print(target)

# Construct Decision Tree Model and test it
from sklearn import tree
model = tree.DecisionTreeClassifier()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(inputs_final, target, test_size=0.2)
model.fit(X_train, y_train)

print(f"Model's accuracy is {model.score(X_test, y_test)}")

#Do some predictions
model_predict = model.predict([[3,22,7.25,1]])
print(model_predict)

