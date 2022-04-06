# This file will attempt to perform linear regression on a set of data

import numpy as np
import pandas as pd
import matplotlib as plt
    
def GD(X, y, theta, alpha, N):
    # performs gradient descent 
    
    m = len(y)
    # Rewrites X to contain initial values for theta0
    X = [np.ones(m,1), X]
    # Initialise histories of cost function
    J_hist = np.zeros(N,1)
    
    # Computes the cost function
    J = (1/2*m) * np.transpose(X*theta - y) * (X*theta - y)
    # Computes the gradient
    G = (1/m) * np.transpose(X) * (X*theta - y)
    
    # Performing gradient descent
    for j in range(1,N+1):
        # Updating theta
        theta = theta - (alpha * G)
        # Save the cost function values
        J_hist(j) == J
        
    return [theta, J_hist]
        
X = pd.read_csv(data.csv)
