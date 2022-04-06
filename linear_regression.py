# This file will attempt to perform linear regression on a set of data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("machine_learning/data_file/data.csv")
labels = ['Re', 'Cd']

x = X[labels[0]]
x = pd.DataFrame(x)
y = X[labels[1]]
y = pd.DataFrame(y)
y = y.to_numpy()

### Tuning parameters
theta_initial = np.ones([2,1])
alpha = 0.01   # Learning rate / %
N = 100         # Number of gradient iterations

### Cost Function algorithm
m = len(y)
initial_vals = pd.DataFrame(np.ones([m,1]))
x = initial_vals.assign(Re = x)
x = x.to_numpy()
diff = np.zeros(m)

### Gradient Descent algorithm
J_hist = np.zeros([N,1])

for j in range(100):
    for i in range(m):
        diff[i] = np.dot(x[i],theta_initial) - y[i]
    # Calculating the cost function
    J = (1/(2*m)) * np.dot(np.transpose(diff),diff)
    # Calculating the gradient
    G = (1/m) * np.dot(np.transpose(x),diff)
    # Updating theta
    theta_initial = theta_initial.transpose() - (alpha * G)
    theta_initial = theta_initial.transpose()
    # Saving the cost function
    J_hist[j] = J

# Plot of regressed line
x_r = np.linspace(-1,7,100)
y_r = pd.DataFrame(np.ones([len(x_r),1]))
y_r = y_r.assign(x = x_r)
y_r = np.dot(y_r.to_numpy(),theta_initial)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.plot(x_r,y_r)
plt.show()

