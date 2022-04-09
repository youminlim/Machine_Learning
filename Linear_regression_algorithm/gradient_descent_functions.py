# This module will be store all functions required to obtain a linear regression for a set of 2 dimensional data

#%% Importing modules required for functions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Functions
    #%%% Data Configuration Function
def config_data(csv_file, labels):
    # This configures the 2 dimensional data for use in the gradient descent and plotting function
    # Reading the data
    df = pd.read_csv(csv_file)
    
    # Separating the data
    x = df[labels[0]]
    x = np.array(x)
    y = df[labels[1]]
    y = np.array(y)
    
    return [x, y]

    #%%% Gradient Descent Function
def GD(X, y, theta, alpha, N):
    # The output theta should be used to calculate the start and end points of the regression line
    # X and y has to be numpy arrays
    
    m = len(y)
    # Rewrites X to contain initial values for theta0
    initial_vals = pd.DataFrame(np.ones([m,1]))
    
    for k in range(X.shape[1]):
        name = ['theta ' + str(k+1)]
        initial_vals = initial_vals.assign(name = X[k])
        
    X = initial_vals.to_numpy()
    print(X)
    
    # Initialise histories of cost function and diff
    J_hist = np.zeros([N,1])
    diff = np.zeros(m)
    
    for j in range(N):
        for i in range(m):
            diff[i] = np.dot(X[i],theta) - y[i]
        # Calculating the cost function
        J = (1/(2*m)) * np.dot(np.transpose(diff),diff)
        # Calculating the gradient
        G = (1/m) * np.dot(np.transpose(X),diff)
        # Updating theta
        theta = theta.transpose() - (alpha * G)
        theta = theta.transpose()
        # Saving the cost function
        J_hist[j] = J
        
    return [theta, J_hist]

    #%%% Plotting Function
def plot_result(x, y, theta, labels):
    [minimum, maximum] = [min(x), max(x)]
    x_r = np.linspace(minimum, maximum, 100)
    y_r = pd.DataFrame(np.ones([len(x_r),1]))
    y_r = y_r.assign(X = x_r)
    y_r= np.dot(y_r.to_numpy(),theta)
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.plot(x_r, y_r)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title("Plot of " + labels[0] + " against " + labels[1])
    ax.legend(["Actual Data", "Regression Fit"])
    plt.show()
