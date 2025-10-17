from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
import pandas as pd 

# DF
digits  = load_digits()

print(dir(digits))

df = pd.DataFrame(digits.data, columns=digits.feature_names)
df['target'] = digits.target



# Test and Train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)

from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1,10,20],
        'kernel': ['rbf','linear']
    }, 
    cv=5, 
    return_train_score=False, 
    n_iter=5
)
rs.fit(digits.data, digits.target)
df1 = pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]
# print(df1)

# Different Models
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'criterion': ['gini', 'entropy'],
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(),
        'params': {
            'C': [1,5,10],
            'solver': ['lbfgs', 'liblinear'],
            'multi_class': ['auto', 'ovr']
        }
    },
    'multinomial' : {
        'model': MultinomialNB(),
        'params': {
            'alpha': [1, 5, 10] # no need to add
        }
    },
    'gaussian' : {
        'model': GaussianNB(),
        'params' : {
            'var_smoothing': [1e-9, 1e-8, 1e-7] # no need to add
        }
    },
    'decision_tree' : {
        'model': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy']
            
        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=3, return_train_score=False)
    clf.fit(digits.data, digits.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df_final = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df_final)