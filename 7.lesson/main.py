import pandas as pd 
import numpy as np 
from sklearn.datasets import load_iris
import seaborn as sns 
from matplotlib import pyplot as plt

#Load dataset
iris = load_iris()
print(dir(iris))
print(iris.feature_names)
print(iris.target_names)
#print(iris.target)
#print(iris.data)

#Scatter plot
sns.scatterplot(x=iris.data[:,2], y=iris.data[:,3], hue=iris.target)
plt.legend(labels = iris.target_names)
plt.title("Petal")
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

#Construct Model and Train/Test it
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
model.fit(X_train, y_train)

print(X_train.shape)
print(f"Model's accuracy is {model.score(X_test, y_test)}")
#print(model.predict(iris.data[0:50]))

#Confusion Matrix
y_predicted = model.predict(X_test) #random prediction of category 
print(y_predicted)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)

# Plot a visual representation of confussion matrix
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix of Iris dataset Testing sample")
plt.xlabel('Predicted value')
plt.ylabel('Truth value')
plt.show()