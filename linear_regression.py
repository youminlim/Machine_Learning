# This file will attempt to perform linear regression on a set of data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("machine_learning/data_file/data.csv")
labels = ['Re', 'Cd']

x = X[labels[0]]
x = np.array(x)
y = X[labels[1]]
y = pd.DataFrame(y)
y = y.to_numpy()

### Tuning parameters
theta_initial = np.ones([2,1])
alpha = 0.11   # Learning rate / %
N = 100         # Number of gradient iterations

[theta, J_hist] = GD(x, y, theta_initial, alpha, N)

# Plot of regressed line
x_r = np.linspace(-1,7,100)
y_r = pd.DataFrame(np.ones([len(x_r),1]))
y_r = y_r.assign(x = x_r)
y_r = np.dot(y_r.to_numpy(),theta)

fig, ax = plt.subplots()
ax.plot(x,y)
ax.plot(x_r,y_r)
plt.show()

