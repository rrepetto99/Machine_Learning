import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
import seaborn as sns 
from matplotlib import pyplot as plt

# Load data set and create a Data frame
iris = load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target 

df['flower_name'] = df['target'].apply(lambda x: iris.target_names[x])

print(df.head())

#Plot some relationships
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='flower_name')
#plt.show()


#Train and the model and prediction
X = df.drop(['target', 'flower_name'], axis='columns')
y = df['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model's accuracy is {accuracy}")

# Create a confusion matrix
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

sns.heatmap(cm, annot=True)
plt.xlabel("Predicted values")
plt.ylabel("Truth values")
plt.show()

