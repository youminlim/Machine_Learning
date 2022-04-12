#%% This will be a module to store the functions to be used in the K-means algorithm

# ============================================================================= 
#     Definitions:
#         
#     centroids: numpy array of shape (K, N), where K is the 
#     number of centroids and N is the dimensions of the points. 
#     
#     points: numpy array of shape (M, N), where M is the 
#     number of points and N is the dimensions of the points.
#     
#     cluster_assignments: numpy array of shape (M,) where 
#     each entry corresponds to an integer (type = int) from 
#     0 to K-1 corresponding to the nearest centroid of a given 
#     point.
# 
#     newCentroids: numpy array of shape (K, N), where K is the 
#     number of centroids, containing the updated centroids 
#     
#     total_error: the sum of the squared distance between all points and 
#     their assigned centroids. (double) 
# =============================================================================
#%% Functions :
    #%%% points2centroids
def points2centroids(centroids, points):
    '''
    Determines the centroid to which each point is nearest, and
    store this as an integer. Assigns each point an identifier 
    corresponding to the centroid it is nearest to.
        
    Inputs: (centroids, points)

    Outputs: cluster_assignments
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

    #%%% recalcCentroids
def recalcCentroids(centroids, points, cluster_assignments):
    '''
    Recalculates centroid locations for each cluster 
    as the mean of locations of all points assigned to it.
    
    Inputs: (centroids, points, cluster_assignments)

    Outputs: newCentroids
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

    #%%% calcError
def calcError(centroids, points, cluster_assignments):
    '''
    Calculates the sum of the squared distance between all points and 
    their assigned centroids.
    
    Inputs: (centroids, points, cluster_assignments)

    Outputs: total_error
    '''
    ###
    M = points.shape[0]
    total_error = 0
    
    for i in range(M):
        total_error += np.linalg.norm(abs(points[i] - centroids[cluster_assignments[i]])**2)    

    return total_error
    
    #%%% Kmeans
def Kmeans(centroids, points, cluster_assignments, max_iter = 30):
    '''
    Performs k-means clustering. Stops when the centroids corrdinates 
    no longer change between iterations or when the maximum number of 
    iterations has been reached.
    
    Inputs: (centroids, points, cluster_assignments)
        
    max_iter = 30 
    '''
    iteration = 0
    mean_error = 0
    
    while iteration < max_iter:
        
        # Cluster points
        cluster_assignments = points2centroids(centroids, points)
        
        # Recalculate centroids
        centroids = recalcCentroids(centroids, points, cluster_assignments)
        
        # Calculate error
        previous_error = mean_error
        error = calcError(centroids, points, cluster_assignments)
        mean_error = error / len(points)
        print_statement = "Error = " + str(mean_error) 
        print(print_statement)
        
        if previous_error == mean_error:
            break
        
        iteration += 1
        
    return (centroids, cluster_assignments, iteration, mean_error)