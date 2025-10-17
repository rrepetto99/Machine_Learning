from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

# Data sample
iris = load_iris()

# Train models
from sklearn.model_selection import train_test_split
X_test, X_train, y_test, y_train = train_test_split(iris.data, iris.target, test_size=0.2)

# Logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# SVM
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# Cross Val validation
from sklearn.model_selection import cross_val_score

# LR
lr_score = cross_val_score(LogisticRegression(), iris.data, iris.target, cv=3)
print(f"Scores for testing multiple times: {lr_score}\n")
print(f"Average Score for Logistic Regression is {np.mean(lr_score)}\n")

# DT
dt_score = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target, cv=3)
print(f"Scores for testing multiple times: {dt_score}\n")
print(f"Average Score for Decision Tree is {np.mean(dt_score)}\n")

# SVM
svm_score = cross_val_score(SVC(), iris.data, iris.target, cv=3)
print(f"Scores for testing multiple times: {svm_score}\n")
print(f"Average Score for SVM is {np.mean(svm_score)}\n")

# RF
rf_score = cross_val_score(RandomForestClassifier(n_estimators=40), iris.data, iris.target, cv=3)
print(f"Scores for testing multiple times: {rf_score}\n")
print(f"Average Score for random forest is {np.mean(rf_score)}")
