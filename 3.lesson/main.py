## Gradient Descent 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

# Simple linear regression
def predict_using_sklearn():
    df = pd.read_csv("3.lesson/test_scores.csv")
    reg = LinearRegression()
    reg.fit(df[['math']], df[['cs']])
    return reg.coef_, reg.intercept_
    
# Gradient Descent
def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 1000000
    n = len(x)
    learning_rate = 0.0002
    
    cost_before = 0
    
    for i in range(iterations):
        y_predicted = x*m_curr + b_curr
        #cost function
        cost = 1/n * sum([val**2 for val in (y-y_predicted)])
        # Caclulate partial derivative for m and b
        m_der = -(2/n)*sum(x*(y-y_predicted))
        b_der = -(2/n)*sum(y-y_predicted)
        #Calculate m_curr and b_curr
        m_curr = m_curr - learning_rate*m_der
        b_curr = b_curr - learning_rate*b_der
        if math.isclose(cost, cost_before, rel_tol=1e-20):
            break
        cost_before = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr,cost, i))
    
    return m_curr, b_curr
        
if __name__ == "__main__":
    df = pd.read_csv("3.lesson/test_scores.csv")
    x = np.array(df['math'])
    y = np.array(df['cs'])

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn,b_sklearn))
    
    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(np.round(m,5), np.round(b,5)))
    