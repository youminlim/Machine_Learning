# This file will attempt to perform linear regression on a set of data

#%% Import modules
import numpy as np
import pandas as pd
import gradient_descent_functions as func

#%% Main program
labels = ['Re', 'Cd']
[x, y] = func.config_data("data_file/data.csv", labels)

    #%%% Linear Regression
# Tuning Parameters
theta_initial = np.ones([2,1])
alpha = 0.11    # Learning rate / %
N = 100         # Number of gradient iterations

[theta, J_hist] = func.GD(x, y, theta_initial, alpha, N)

    #%%% Multivariable Linear Regression
polynomial_order = 3
x_mult = np.zeros([len(x),polynomial_order])
x_mult = pd.DataFrame(x_mult)

for k in range(1,polynomial_order+1):
    x_mult[k-1] = np.power(x,k)
    
    x_mult = pd.DataFrame (x_mult)
    
# Tuning Parameters
theta_initial = np.ones([polynomial_order+1,1])
alpha = 0.0001
N = 10000

#[theta_multi, J_hist_multi] = func.GD(x_mult, y, theta_initial, alpha, N)

m = len(y)
# Rewrites X to contain initial values for theta0
initial_vals = pd.DataFrame(np.ones([m,1]))

header = np.chararray(x_mult.shape[1]+1)
header[0] = 'theta 0'

for m in range(x_mult.shape[1]):
    header[m] = 'theta ' + str(m+1)
    
print(header)


    
x_mult = initial_vals.to_numpy()

# Initialise histories of cost function and diff
J_hist = np.zeros([N,1])
diff = np.zeros(m)

for j in range(N):
    for i in range(m):
        diff[i] = np.dot(x_mult[i],theta_initial) - y[i]
    # Calculating the cost function
    J = (1/(2*m)) * np.dot(np.transpose(diff),diff)
    # Calculating the gradient
    G = (1/m) * np.dot(np.transpose(X),diff)
    # Updating theta
    theta_initial = theta_initial.transpose() - (alpha * G)
    theta_initial = theta_initial.transpose()
    # Saving the cost function
    J_hist[j] = J
    
    #%%% Plot of regressed line
func.plot_result(x_mult, y, theta_multi, labels)

