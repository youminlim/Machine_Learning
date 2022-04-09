# This file will attempt to perform linear regression on a set of data

#%% Import modules
import numpy as np
import gradient_descent_functions as func

#%% Main program
labels = ['Re', 'Cd']
[x, y] = func.config_data("data_file/data.csv", labels)

    #%%% Tuning parameters
theta_initial = np.ones([2,1])
alpha = 0.11    # Learning rate / %
N = 100         # Number of gradient iterations

[theta, J_hist] = func.GD(x, y, theta_initial, alpha, N)

    #%%% Plot of regressed line
func.plot_result(x, y, theta, labels)