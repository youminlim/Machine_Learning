# K means algorithm

#%% Import libraries
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import Kmeans_functions as func

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
centroids = points[0:11,:]
cluster_assignments = points2centroids(centroids, points)
newCentroids = recalcCentroids(centroids, points, cluster_assignments)
total_error = calcError(centroids, points, cluster_assignments)

centroids, cluster_assignments, iteration, mean_error = Kmeans(centroids, points, cluster_assignments)

#%%%
#SELF-TEST
K = 5  # Input the number of clusters (centroids) to compute

#Initialize centroids to random data points
M = points.shape[0] # number of points
indices = []
while len(indices) < K:
    index = np.random.randint(0, M)
    if not index in indices: #making sure each centroid is unique (i.e. no two centroids have the same initial coordinates)
        indices.append(index)
        
initialCentroids = points[indices,:]
cluster_assignments=assignPointsToCentroids(initialCentroids, points)
max_iterations = 300
#perform K-means on data
centroids, cluster_assignments, iteration, total_error = myKmeans(initialCentroids, points,cluster_assignments,max_iterations)

import matplotlib.pyplot as plt
plt.scatter(points[:, 0], points[:, 1], c = cluster_assignments);
plt.plot(centroids[:,0],centroids[:,1],'bo'); 