import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt 
import numpy as np 

# Import dataframe
df = pd.read_csv("Logistic_Regression/HR_comma_sep.csv")
print(df.head(10))

# Some summary statistics

left = df.loc[df['left'] == 1]
print(left)
print(left.shape)

#Filter data
df_new = df.iloc[:,:8]
print(df_new)

df_grouped = df_new.groupby('left').mean()
print(df_grouped)


# Plot bar charts showing impact of employee salaries on retention
sns.set_theme(style="whitegrid")
sns.countplot(data=df, x="salary", hue="left", alpha=0.6)
plt.ylabel('N')
plt.show()

# Plot bar charts showing corelation between department and employee retention
sns.countplot(data=df, x="Department", hue="left", alpha=0.6)
plt.ylabel('N')
plt.show()


# Now build logistic regression model using variables that were narrowed down in step 1

df_sub = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]


# convert salaries to dummy variables
dummies = pd.get_dummies(df_sub[['salary']], dtype=int)
df_filtered = df_sub.drop(['salary'], axis='columns')
df_merged_with_dummies = pd.concat([df_filtered, dummies], axis='columns')
print(df_merged_with_dummies)

# Model
from sklearn.model_selection import train_test_split
X = df_merged_with_dummies
y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)
#model.predict_proba(X_test) #probability

# Accuracy
accuracy = model.score(X_test,y_test)
print(f"Model's accuracy is {np.round(accuracy, 4)}")
