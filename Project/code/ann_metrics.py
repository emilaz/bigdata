import numpy as np
from scipy.io import loadmat
import time






def linear_search(query, data, k, norm=2, print_status=False):
    '''
    Compute true nearest neighbors of every query point.

    Input:
        query: data points to query (full precision)
        size(query) = (nQueries, nFeatures)
        data: dataset to query (full precision)
        size(data) = (nData, nFeatures)
        k: number of neighbors to return
        norm: norm to be used to compare features

    Output:
        neighbors: k closest neighbors of each data point
        size(neighbors) = [nQueries, k]
        distances: distances to each neighbor
        size(distances) = [nQueries, k]
    '''

    [nQueries, nFeatures] = query.shape
    [nData, nFeatures] = data.shape
    times = np.zeros([nQueries, 1])
    neighbors = np.zeros([nQueries, k]).astype(int)
    distances = 100000*np.ones([nQueries, k])
    max_dist = 100000*np.ones([nQueries, 1])
    max_dist_idx = np.zeros([nQueries, k]).astype(int)

    for i in range(nQueries):
        start_time = time.time()
        for j in range(nData):
            dist = np.linalg.norm(query[i,:] - data[j,:], ord=norm)
            if j < k:
                distances[i, j] = dist 
                neighbors[i, j] = j
                max_dist[i] = np.max(distances[i, :])
                max_dist_idx[i] = np.argmax(distances[i, :])
            elif dist < max_dist[i]:
                distances[i, max_dist_idx[i]] = dist 
                neighbors[i, max_dist_idx[i]] = j
                max_dist[i] = np.max(distances[i, :])
                max_dist_idx[i] = np.argmax(distances[i, :])

        sort_idx = np.argsort(distances[i, :])
        dist_sorted = np.zeros([1,k])
        idx_sorted = np.zeros([1,k])
        for j in range(k):
            dist_sorted[0,j] = distances[i, sort_idx[j]]
            idx_sorted[0,j] = neighbors[i, sort_idx[j]]
        stop_time = time.time()

        times[i] = stop_time - start_time
        neighbors[i, :] = idx_sorted
        distances[i, :] = dist_sorted

        if print_status:
            print(str(i) + '/' + str(nQueries))

    return (neighbors, distances, times)




def recall_and_search(query, data, ann, norm=2, query_idx=-1):
    '''
    Calculate recall for query (without ground truth)

    Input:
        query: data points to query (full precision)
        size(query) = (nQueries, nFeatures)
        data: dataset to query (full precision)
        size(data) = (nData, nFeatures)
        ann: approximate nearest neighbors returned by other method
        size(ann) = (nQueries, R)
        norm: norm to be used to compare features
        query_idx: index of queries if they are in dataset

    Output:
        recall: percent of queries that had true nearest neighbor in returned queries
    '''
    if query_idx == -1:
        neighbors, distances = linear_search(query, data, 1, norm=norm)
        [nQueries, nFeatures] = query.shape
        correct = 0
        for i in range(nQueries):
            if neighbors[i,0] in ann[i,:]:
                correct += 1
    else:
        neighbors, distances = linear_search(query, data, 2, norm=norm)
        [nQueries, nFeatures] = query.shape
        correct = 0
        for i in range(nQueries):
            if neighbors[i,0] == query_idx[i]:
                if neighbors[i,1] in ann[i,:]:
                    correct += 1
            else:
                if neighbors[i,0] in ann[i,:]:
                    correct += 1

    return float(correct) / float(nQueries)





def recall(ann, gt):
    '''
    Calculate recall for query (with ground truth)

    Input:
        ann: approximate nearest neighbors returned by other method
        size(ann) = (nQueries, R)
        gt: true k nearest neighbors of each query point (in order)
        size(gt) = (nQueries, k)
    '''
    [nQueries, R] = ann.shape
    correct = 0

    for i in range(nQueries):
        if gt[i,0] in ann[i,:]:
            correct += 1

    return float(correct) / float(nQueries)






## Code to calculate true nearest neighbors of ANN 
# data = loadmat('X_train_small.mat')
# d = len(data['X_train'][0])
# n = len(data['X_train'])
# names = data['names']
# classes = data['class_num']

# X_train = np.zeros((n,d)).astype('float32')
# for i in range(n):
#     X_train[i,:] = data['X_train'][i]




# data_test = loadmat('X_test_small.mat')
# d_test = len(data_test['X_test'][0])
# n_test = len(data_test['X_test'])
# names_test = data_test['names_test']
# classes_test = data_test['class_num_test']

# X_test = np.zeros((n_test,d_test)).astype('float32')
# for i in range(n_test):
#     X_test[i,:] = data_test['X_test'][i]



# neighbors, distances, times = linear_search(X_test, X_train, 10, print_status=True)
# np.savez('imagenet_truth_small',neighbors,distances, times)





# test linear search
# data = np.random.randn(100,10)
# neighbors, distances = linear_search(data, data, 5)
# print(neighbors[0,0],neighbors[0,1],neighbors[0,2])
# print(distances[0,0],distances[0,1],distances[0,2])
# print(neighbors[1,0],neighbors[1,1],neighbors[1,2])
# print(distances[1,0],distances[1,1],distances[1,2])
# print(neighbors[2,0],neighbors[2,1],neighbors[2,2])
# print(distances[2,0],distances[2,1],distances[2,2])


