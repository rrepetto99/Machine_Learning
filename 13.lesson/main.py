import pandas as pd 
import numpy as np 
from sklearn.datasets import load_wine

#Dataset
wine = load_wine()

#print(wine.target_names)

df = pd.DataFrame(wine.data, columns=wine.feature_names)

df['target'] = wine.target 

# Check whether there are empty values
#print(df.columns[df.isna().any()])

# Assign variables
X = df.drop(['target'], axis='columns')
y = df['target']

# Model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Gaussian
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(X_train, y_train)

#print(f"Accuracy for Gaussian classfier is {model.score(X_test, y_test)}")

#print(y_test[:10])
#print(model.predict(X_test[:10]))
#print(model.predict_proba(X_test[:10])) #predict prbabilit for each wine class

from sklearn.model_selection import cross_val_score
cross_Gauss = cross_val_score(GaussianNB(), X_train, y_train, cv=5)
print(f"Gaussian: accuracy in multiple testings: {cross_Gauss}\n")
print(f"Gaussian: average accuracy in multiple testings: {np.mean(cross_Gauss)}\n")

# Multinominal
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train,y_train)

cross_Multi = cross_val_score(MultinomialNB(), X_test, y_test, cv=5)
print(f"Multinominal: accuracy in multiple testings: {cross_Multi}\n")
print(f"Multinominal: average accuracy in multiple testings: {np.mean(cross_Multi)}\n")