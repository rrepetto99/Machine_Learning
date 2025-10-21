import pandas as pd 
import numpy as np 
from sklearn.datasets import load_digits
import seaborn as sns 
from matplotlib import pyplot as plt

#Load dataset and adjust it
digits = load_digits()
print(dir(digits))
df = pd.DataFrame(digits.data, columns=digits.feature_names)

#Add target's column which is y variable
df['target'] = digits.target 

# Construct a model and train/test it
from sklearn.model_selection import train_test_split

X = df.drop(['target'], axis='columns')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(len(X_test))

# Support Vector Machine
from sklearn.svm import SVC
model = SVC()

model.fit(X_train, y_train)

accuracy = model.score(X_test,y_test)
print(f"Model's accuracy is {accuracy}") # Around 0.9833,max 0.9916

# Regularization (C)
model_C = SVC(C=6)
model_C.fit(X_train, y_train)
accuracy_C = model_C.score(X_test, y_test)
print(f"Model's accuracy is {accuracy_C}") # Around 0.9888 with C=3 , max 0.9944

# Gamma 
model_G = SVC(gamma=1)
model_G.fit(X_train, y_train)
accuracy_G = model_G.score(X_test, y_test)
print(f"Model's accuracy is {accuracy_G}") # Max 0.0833

# Kernel
model_K = SVC(kernel='rbf')
model_K.fit(X_train, y_train)
accuracy_K = model_K.score(X_test, y_test)
print(f"Model's accuracy is {accuracy_K}") # Max 0.9833 with linear, with rbf max is 0.9944
