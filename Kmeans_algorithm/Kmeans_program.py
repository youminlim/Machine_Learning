# K means algorithm

#%% Import libraries
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt

#%% Main Program
    #%%% Load data
points = loadtxt('data_file/Kmeans_data.csv', delimiter=',')
length = points.shape[0]
x = np.zeros(length)
y = np.zeros(length)

for i in range(length):
    [x[i], y[i]] = points[i]
    
    #%%% Plot data
fig, ax = plt.subplots()
ax.scatter(x, y, marker='*')
plt.show()

    #%%% Load model components
def assignPointsToCentroids(centroids, points):
    '''
    Determines the centroid to which each point is nearest, and
    store this as an integer.
        
    Inputs:
        centroids: numpy array of shape (K, N), where K is the 
        number of centroids and N is the dimensions of the points. 
        
        points: numpy array of shape (M, N), where M is the 
        number of points and N is the dimensions of the points.

    Outputs:
        cluster_assignments: numpy array of shape (M,) where 
        each entry corresponds to an integer (type = int) from 
        0 to K-1 corresponding to the nearest centroid of a given 
        point.
    '''
    K = centroids.shape[0]
    M = points.shape[0]
    
    cluster_assignments = np.zeros(M, dtype = int)
    distance = [[]] * K
            
    for i in range(M):
        for j in range(K):
            distance[j] = np.linalg.norm(points[i] - centroids[j])
        cluster_assignments[i] = int(distance.index(min(distance)))
        
    return cluster_assignments

def recalcCentroids(centroids, points, cluster_assignments):
    '''
    Recalculates centroid locations for each cluster 
    as the mean of locations of all points assigned to it.
    
    Inputs:
        centroids: numpy array of shape (K, N), where K is the 
        number of centroids and N is the dimensions of the points. 
        
        points: numpy array of shape (M, N), where M is the 
        number of points and N is the dimensions of the points.
        
        cluster_assignments: numpy array of shape (M,) where 
        each entry corresponds to an integer (type = int) from 
        0 to K-1 corresponding to the nearest centroid of a given 
        point.

    Outputs:
        newCentroids: numpy array of shape (K, N), where K is the 
                   number of centroids, containing the updated 
                   centroids 
    '''
    K = centroids.shape[0]
    M = points.shape[0]
    
    cluster_points = np.array([])
    newCentroids = np.array([])
    counter = 0
    
    # Separating each cluster's set of points
    for i in range(K):
        for j in range(M):
            if cluster_assignments[j] == i:
                cluster_points = np.append(cluster_points, points[i])
                counter += 1
        cluster_points = np.reshape(cluster_points, (counter, 2))
        # Calculating the mean between all points in a cluster
        newCentroids = np.append(newCentroids, np.mean(cluster_points, axis = 0))
    
    newCentroids = np.reshape(newCentroids, (K,2))

    return newCentroids

def calcError(centroids, points, cluster_assignments):
    '''
    Calculates the sum of the squared distance between all points and 
    their assigned centroids.
    
    Inputs:
        centroids: numpy array of shape (K, N), where K is the 
        number of centroids and N is the dimensions of the points. 
        
        points: numpy array of shape (M, N), where M is the 
        number of points and N is the dimensions of the points.
        
        cluster_assignments: numpy array of shape (M,) where 
        each entry corresponds to an integer (type = int) from 
        0 to K-1 corresponding to the nearest centroid of a given 
        point.

    Outputs:
        total_error: the sum of the squared distance between all points and 
        their assigned centroids. (double) 
    
    '''
    ###
    K = centroids.shape[0]
    M = points.shape[0]
    
    error = cluster_assignments
    
    for i in range(M):
        error[i] = np.linalg.norm(points[i] - centroids[cluster_assignments[i]])
    total_error = sum(error)
    ###
    return total_error   

#centroids = np.array([[1,2],[0.5,0.5]])
centroids = points[0:11,:]
cluster_assignments = assignPointsToCentroids(centroids, points)
newCentroids = recalcCentroids(centroids, points, cluster_assignments)
total_error = calcError(centroids, points, cluster_assignments)
#%%
def myKmeans(centroids, points, cluster_assignments, max_iter = 30):
    '''
    Performs k-means clustering. Stops when the centroids corrdinates 
    no longer change between iterations or when the maximum number of 
    iterations has been reached.
    
    Inputs:
        centroids: numpy array of shape (K, N), where K is the 
        number of centroids and N is the dimensions of the points. 
        
        points: numpy array of shape (M, N), where M is the 
        number of points and N is the dimensions of the points.
        
        cluster_assignments: numpy array of shape (M,) where 
        each entry corresponds to an integer (type = int) from 
        0 to K-1 corresponding to the nearest centroid of a given 
        point.
        
        max_iter = 30
        
    '''
    ###
    
    ###
    return (centroids, cluster_assignments, iteration, mean_error)

    
    cluster_points = np.array([])
    counter = 0
    total_error = np.zeros(K)
    # Separating each cluster's set of points
    for i in range(K):
        for j in range(M):
            if cluster_assignments[j] == i:
                cluster_points = np.append(cluster_points, points[i])
                counter += 1
        cluster_points = np.reshape(cluster_points, (counter, 2))
        
        a = np.ones(counter)

        total_error[i] = np.sqrt(sum((cluster_points - centroids[i])**2))