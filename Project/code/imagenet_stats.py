import numpy as np 
from scipy.io import loadmat
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import faiss
import sh_wrapper as sh
import ann_metrics
import lsh



def get_imagenet_neighbors(neighbors,names,query_id, k):
	images = []

	for i in range(len(neighbors)):
		nidx = neighbors[i]
		name_split = names[nidx][0][0].split('_')
		im_url = ''

		with open('./imagenet_sift/image_urls/' + name_split[0] + '.txt') as fp:  
			line = fp.readline()
			while line:
		   		name_url = line.split(' ')
		   		if name_url[0] == names[nidx][0][0]:
		   			im_url = name_url[1]
		   			break
		   		line = fp.readline()

			try:
				response = requests.get(im_url)
				img = Image.open(BytesIO(response.content))
				images.append(img)
			except:
				print('bad url')

		if len(images) == k:
			break

	name_split = names[query_id][0][0].split('_')
	im_url = ''

	with open('./imagenet_sift/image_urls/' + name_split[0] + '.txt') as fp:  
	   line = fp.readline()
	   while line:
	   		name_url = line.split(' ')
	   		if name_url[0] == names[query_id][0][0]:
	   			im_url = name_url[1]
	   			break
	   		line = fp.readline()
	response = requests.get(im_url)
	img = Image.open(BytesIO(response.content))
	images.append(img)

	return images





##################################################################################################
# load data
data = loadmat('X_train_small.mat')
d = len(data['X_train'][0])
n = len(data['X_train'])
names = data['names']
classes = data['class_num']

X_train = np.zeros((n,d)).astype('float32')
for i in range(n):
	X_train[i,:] = data['X_train'][i]




data_test = loadmat('X_test_small.mat')
d_test = len(data_test['X_test'][0])
n_test = len(data_test['X_test'])
names_test = data_test['names_test']
classes_test = data_test['class_num_test']

X_test = np.zeros((n_test,d_test)).astype('float32')
for i in range(n_test):
	X_test[i,:] = data_test['X_test'][i]




gt = np.load('imagenet_truth_small.npz')
gt_neighbors = gt['arr_0']
gt_distances = gt['arr_1']
linear_compute_times = gt['arr_2']
##################################################################################################



nn = 100 # number of nearest neighbors to search for
RUN_PQ = True
RUN_SH = True
RUN_LSH = False
RUN_LSH2 = True


##################################################################################################
# run with PQ
'''
nlist is the number of centroids (regions) the data is partitioned into
m is number of subquantizers
nbits is the number of bits per subquantizer 
I believe there are m*nbits bits used in total (per vector)
decreasing nlist seems to increase the accuracy (at the expense perhaps of computation time)
increasing m increases the accuracy
'''


if RUN_PQ:
	m = 40 # bytes/vector
	nlist = 1
	nbits = 6 # number of bits per subvector
	quantizer = faiss.IndexFlatL2(d)
	print('RUNNING PQ')
	print('initializing')
	index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
	print('training')
	index_ivfpq.train(X_train)
	print('adding')
	index_ivfpq.add(X_train)                 
	print('searching')
	D, I = index_ivfpq.search(X_test, nn) # I contains indices
	print(I.shape)

	recall_pq = ann_metrics.recall(I, gt_neighbors)
##################################################################################################




##################################################################################################
# run with SH
'''
nbits (second argument to trainSH) is the total number of bits used to represent data point
the vector in B for each data point will be a vector of 8 bit ints with length nbits / 8
SHparam contains 4 elements:
	pc: nFeatures x nBits (float64)
	mn: nBits x 1 (float64)
	mx: nBits x 1 (float64)
	modes: nBits x nBits (float64, though it seems to be actually a binary matrix)
'''

if RUN_SH:
	nbits = 256
	print('\n\nRunning SH')
	print('training')
	SHparam = sh.trainSH(X_train, nbits)
	print('compressing')
	[B,U] = sh.compressSH(X_train, SHparam)
	print('searching')
	neighbors_sh = sh.get_neighbors(B, X_test, SHparam, nn)
	print(neighbors_sh.shape)

	recall_sh = ann_metrics.recall(neighbors_sh, gt_neighbors)
##################################################################################################





##################################################################################################
# run with LSH
'''
uses L*k bits to represent each point
L is number of hash functions (bins?), k is number of bits per hash function
'''

if RUN_LSH:
	k = 10
	L = 25
	print('\n\nRunning LSH')
	functions, hashed_A = lsh.lsh_setup(X_train, k=k, L=L)
	print('searching')
	lsh_out = lsh.lsh_search2(X_test, hashed_A, X_train, functions, num_neighbors=nn)
	print(lsh_out.shape)

	recall_lsh = ann_metrics.recall(lsh_out, gt_neighbors)
##################################################################################################





##################################################################################################
# run with LSH Faiss
'''
nbits is total number of bits used for each point?
'''

if RUN_LSH2:
	nbits = 256
	print('\n\nRunning FAISS LSH')
	index_lsh = faiss.IndexLSH(d, nbits)
	index_lsh.train(X_train)
	index_lsh.add(X_train)
	D, I = index_lsh.search(X_test, nn)
	print(I.shape)

	recall_lsh2 = ann_metrics.recall(I, gt_neighbors)
##################################################################################################






print('\n\n\n')
if RUN_LSH:
	print('LSH = ' + str(recall_lsh))
if RUN_SH:
	print('SH = ' + str(recall_sh))
if RUN_PQ:
	print('PQ = ' + str(recall_pq))
if RUN_LSH2:
	print('Faiss LSH = ' + str(recall_lsh2))







'''
x['X_train'][i] is ith feature vector
x['names'][i][0][0] is the ith name
'''