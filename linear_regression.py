# This file will attempt to perform linear regression on a set of data

#%% Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Define Functions
def config_data(csv_file, labels):
    # Reading the data
    df = pd.read_csv(csv_file)
    
    # Separating the data
    x = df[labels[0]]
    x = np.array(x)
    y = df[labels[1]]
    y = np.array(y)
    
    return [x, y]

def GD(X, y, theta, alpha, N):
    # performs gradient descent 
    # X and y has to be numpy arrays
    
    m = len(y)
    # Rewrites X to contain initial values for theta0
    initial_vals = pd.DataFrame(np.ones([m,1]))
    X = initial_vals.assign(x = X)
    X = X.to_numpy()
    
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
    plt.show()
#%% Main program
labels = ['Re', 'Cd']
[x, y] = config_data("data_file/data.csv", labels)

#%%% Tuning parametersd
theta_initial = np.ones([2,1])
alpha = 0.11   # Learning rate / %
N = 100         # Number of gradient iterations

[theta, J_hist] = GD(x, y, theta_initial, alpha, N)

#%%% Plot of regressed line
plot_result(x, y, theta, labels)