import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np 

# Load dataset
df = pd.read_csv("17.lesson/heart.csv")
print(df.shape)

cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']

# Compute z-scores only for these columns
z_scores = (df[cols] - df[cols].mean()) / df[cols].std()

# Remove outliers
df_no_outliers = df[(np.abs(z_scores) <= 3).all(axis=1)]

print(df_no_outliers)
   
# X = df.drop(['HeartDisease'], axis=1)
# print(X)
# y = df['HeartDisease']

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

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X, y)
# print(model.score(X,y))


# Apply Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

#LR
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(f"SCA of LR: {model.score(X_test, y_test)}\n")

#SVM
from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train)
print(f"SCA of SVM: {model.score(X_test,y_test)}\n")

# RF
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
print(f"SCA of RF: {model.score(X_test,y_test)}\n")

# Classification
from sklearn.model_selection import GridSearchCV
model_params = {
    'svm': {
        'model': svm.SVC(),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(),
        'params': {
            'solver': ['liblinear', 'lbfgs'],
            'C': [1,5,10]
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(X_scaled, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df_final = pd.DataFrame(scores,columns=['model','best_score','best_params'])
#print(df_final)


# Use PCA to reduce dimensions
from sklearn.decomposition import PCA

pca = PCA(0.95)
X_pca = pca.fit_transform(X)

#Check explainedvariance
pca.explained_variance_ratio_
pca.n_components_
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)

#LR
model = LogisticRegression()
model.fit(X_train_pca, y_train)
print(f"PCA of LR: {model.score(X_test_pca, y_test)}\n")

#SVM
model = svm.SVC()
model.fit(X_train_pca, y_train)
print(f"PCA of SVM: {model.score(X_test_pca,y_test)}\n")

# RF
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train_pca, y_train)
print(f"PCA of RF: {model.score(X_test_pca,y_test)}\n")