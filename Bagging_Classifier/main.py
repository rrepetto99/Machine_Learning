import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np 

# Load dataset
df = pd.read_csv("Bagging_Classifier/heart.csv")
print(df.shape)

cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']

# Compute z-scores only for these columns
z_scores = (df[cols] - df[cols].mean()) / df[cols].std()

# Remove outliers
df_no_outliers = df[(np.abs(z_scores) <= 3).all(axis=1)]

#print(df_no_outliers)

# Assign dummy variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dfle = df_no_outliers
dfle.loc[:, 'Sex'] = le.fit_transform(dfle['Sex'])
dfle.loc[:, 'ChestPainType'] = le.fit_transform(dfle['ChestPainType'])
dfle.loc[:, 'RestingECG'] = le.fit_transform(dfle['RestingECG'])
dfle.loc[:, 'ExerciseAngina'] = le.fit_transform(dfle['ExerciseAngina'])
dfle.loc[:, 'ST_Slope'] = le.fit_transform(dfle['ST_Slope'])

X = dfle.drop(['HeartDisease'], axis=1).values
y = dfle['HeartDisease'].values


# ML
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled[:3])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=20)


#SVM
from sklearn import svm
from sklearn.model_selection import cross_val_score
# model = svm.SVC()
# model.fit(X_train, y_train)
# print(f"SCA of SVM: {model.score(X_test,y_test)}")

# Standalone
scores = cross_val_score(svm.SVC(), X, y, cv=5)
print(f"Standalone of SVM: {np.mean(scores)}")

# Bagging
from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(estimator=svm.SVC(), n_estimators=100, max_samples=0.8)
bag_model.fit(X_train, y_train)
scores = cross_val_score(bag_model, X, y , cv=5)
print(f"Bagging of SVM: {np.mean(scores)}\n")



#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
print(f"Standalone of DTC: {np.mean(scores)}")

bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, oob_score=True, random_state=0)
bag_model.fit(X_train, y_train)
bag_model.score(X_test, y_test)
scores = cross_val_score(bag_model, X, y, cv=5)
print(f"Bagging of DTC: {np.mean(scores)}\n")

# Random forest
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print(f"Random forest {np.mean(scores)}")
