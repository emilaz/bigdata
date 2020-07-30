# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    dist=sum(abs(u-v))
    return dist

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions 
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0, 
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold, 
                                   high = max_threshold + 1, 
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets 
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as 
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)
    
    distances = list(map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums))
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]
    print('Datapoints in the same bucket: ', len(distances))

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        print(row_num)
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        print('jo this is row no:', row_num)
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    distances=map (lambda x: (x,l1(A[x],A[query_index])), range(A.shape[0]))
    best = sorted(distances, key=lambda x : x[1])[1:num_neighbors+1]
    return [b[0] for b in best]

# TODO: Write a function that computes the error measure
def error(A,queries,best_lin,best_hash):
    tot = 0
    for idx,query in enumerate(queries):
        lin_sum = sum([l1(A[query],A[v]) for v in best_lin[idx]])
        hash_sum = sum([l1(A[query], A[v]) for v in best_hash[idx]])
        tot += hash_sum/lin_sum
    return tot/10

def get_lin_and_lsh_searches(A, queries,L,k, nn = 3):
    (funcs, hashed_A) = lsh_setup(A, L=L, k=k)
    res_hash = []
    res_lin = []
    for idx in queries:  # get hash results
        while True:
            resh = lsh_search(A, hashed_A, funcs, idx, nn)
            if (len(resh)) == nn:
                break
        res_hash += [resh]
        resl = linear_search(A, idx, nn)
        res_lin += [resl]
    return res_lin,res_hash

def get_lin_and_lsh_times(A, queries,L,k, nn = 3):
    (funcs, hashed_A) = lsh_setup(A, L=L, k=k)
    lsh_time = []
    lin_time = []
    res_lsh = []
    res_lin = []
    for idx in queries:  # get hash results
        start = time.clock()
        res = lsh_search(A, hashed_A, funcs, idx, nn)
        res_lsh +=[res]
        lsh_time += [time.clock() - start]
        start = time.clock()
        res_lin += [linear_search(A, idx, nn)]
        lin_time += [time.clock() - start]
    avg_lsh_time = sum(lsh_time) / len(lsh_time)
    avg_lin_time = sum(lin_time) / len(lin_time)
    err = error(A, queries, res_lin, res_lsh)
    return avg_lin_time,avg_lsh_time, err

def get_errors_for_k(A,queries, k_range):
    errors = []
    for k in k_range:
        print('k=',k)
        res_lin, res_hash = get_lin_and_lsh_searches(A, queries, L=10, k=k)
        err = error(A,queries,res_lin,res_hash)
        errors += [err]
    return errors

def get_errors_for_l(A,queries, L_range):
    errors = []
    for L in L_range:
        print('L=',L)
        res_lin, res_hash = get_lin_and_lsh_searches(A,queries,L,k=24)
        err = error(A, queries, res_lin, res_hash)
        errors += [err]
    return errors


# TODO: Solve Problem 4
def problem4():
    #comparison of both:
    #A = load_data('./data/patches.csv')
    A = np.random.normal(255/2,2,(5000,400))
    g1 = np.random.normal(150/2,2,(5000,400))
    g2 = np.random.normal(300 / 2, 2, (5000, 400))
    chooser = np.random.choice(2,5000).astype('bool')
    B = np.concatenate((g1[chooser],g2[~chooser]))
    np.random.shuffle(B)
    print(B.shape)
    C = np.random.uniform(0, 255, (5000, 400))
    print('Data read')
    queries = list(range(99,1000,100))
    k=500
    L=5
    # l_range = list(range(10,21,2))
    # k_range = list(range(16,25,2))

    #I : Get speeds:
    print('We chose {} AND functions with {} OR functions, amounting to {} hash functions in total'.format(k,L,k*L))
    for K,typ in zip([A,B,C],['Gaussian, STD 2, Mean 127','Mixed Gausian, STD 2, Mean 75 and 150','Uniform']):
        print(typ)
        lin_time,lsh_time, err = get_lin_and_lsh_times(K, queries,L,k,3)
        print('Linear Search time in s:',lin_time)
        print('LSH Search time in s:',lsh_time)
        print('Error:', err)
    # plt.title('Hash retrieval error for L=10')
    # plt.xlabel('k')
    # plt.ylabel('Error')
    # plt.show()

    # #II: Get error plots
    # l_errors = get_errors_for_l(A,queries,l_range)
    # print('Got L errors')
    #
    # plt.plot(l_range,l_errors)
    # plt.title('Hash retrieval error for k=24')
    # plt.xlabel('L')
    # plt.ylabel('Error')
    # plt.show()
    #
    # k_errors = get_errors_for_k(A,queries,k_range)
    #
    # plt.plot(k_range,k_errors)
    # plt.title('Hash retrieval error for L=10')
    # plt.xlabel('k')
    # plt.ylabel('Error')
    # plt.show()

    #III: Get pics
    # lin_res, lsh_res = get_lin_and_lsh_searches(A,[99],10,24,10)
    # #plot linear res
    # plot(A,lin_res[0],'./pics/linear-idx100')
    # # plot lsh res
    # plot(A, lsh_res[0], './pics/lsh-idx100')
    # #plot orig
    # plot(A,[99],'./pics/orig')

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

#     ### TODO: Write your tests here (they won't be graded,
#     ### but you may find them helpful)


if __name__ == '__main__':
 #   unittest.main() ### TODO: Uncomment this to run tests
    problem4()
