#%% Linear Regression Function

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
    diff = np.zeros([N,1])
    
    for j in range(N):
        for i in range(m):
            diff[i] = np.dot(x[i],theta) - y[i]
        # Calculating the cost function
        J = (1/(2*m)) * np.dot(np.transpose(diff),diff)
        # Calculating the gradient
        G = (1/m) * np.dot(np.transpose(x),diff)
        # Updating theta
        theta = theta.transpose() - (alpha * G)
        theta = theta.transpose()
        # Saving the cost function
        J_hist[j] = J
        
    return [theta, J_hist]