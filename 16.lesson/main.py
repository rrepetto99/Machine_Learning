import pandas as pd 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import seaborn as sn
digits = load_digits()

# data frame
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target

#Models
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

print(f"Simple KNN accuracy: {knn.score(X_test, y_test)}")

# test different values of KNN
from sklearn.model_selection import GridSearchCV

model_params = {
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 10],
        }
    }
}
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=20, return_train_score=False)
    clf.fit(digits.data, digits.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df_final = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df_final)

# Plot CM
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

# Make Classification Report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))