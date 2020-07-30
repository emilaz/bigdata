#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import itertools
import matplotlib.pyplot as plt


# In[ ]:


#set context
conf = SparkConf()
sc = SparkContext(conf=conf)


# In[ ]:


max_iter = 20
clusters = 10


# In[ ]:


def split(l):
    line=l.split(' ')
    val = [float(s) for s in line if s is not ''] 
    return tuple(val)

def combine_with_cluster(l,clu):
    return [(l,c) for c in clu]

def calc_dist(l, norm=2):
    dist = np.linalg.norm(np.array(l[0])-np.array(l[1]),ord=norm)
    return (l[0],(dist,l[1]))

def calc_new_cluster(l):
    l = list(l)
    size = len(l)
    agg = sum(np.array(l))
    return tuple(agg/size)

def get_costs(init_clusters,norm):
    costs = []
    curr_clusters = init_clusters
    for i in range(max_iter): # k - means algorithm
        combos = docs.flatMap(lambda l:combine_with_cluster(l,curr_clusters)) #get (vec,clu) pairs
        combos_w_euc_dist = combos.map(lambda l:calc_dist(l,norm)) #add the distance to said cluster
        combos_w_euc_dist_grouped = combos_w_euc_dist.groupByKey()
        closest = combos_w_euc_dist_grouped.mapValues(lambda val: min(list(val))) # filter such that only the closest (dist,clust) remains as value for a vector
        costs += [sum(closest.map(lambda l:l[1][0]).collect())] #add costs for this clustering
        curr_clusters = closest.map(lambda l : (l[1][1],l[0])).groupByKey() #sort all vectors to its cluster
        new_clusters = curr_clusters.map(lambda l: calc_new_cluster(l[1])) #calculate new clusters
        curr_clusters = new_clusters.collect()
        print('Current cost is', costs[-1])
    return costs


# In[ ]:


#read file in
docs = sc.textFile('./data/data.txt').map(split)
c1 = sc.textFile('./data/c1.txt').map(split)
c2 = sc.textFile('./data/c2.txt').map(split)


# In[ ]:


cluster_rand = c1.collect()
cluster_max = c2.collect()


# In[ ]:


costs_rand = get_costs(cluster_rand,2)

costs_max = get_costs(cluster_max,2)

# costs_max = []
# curr_clusters = cluster_max
# for i in range(max_iter): # k - means algorithm
#     combos = docs.flatMap(lambda l:combine_with_cluster(l,curr_clusters)) #get (vec,clu) pairs
#     combos_w_euc_dist = combos.map(lambda l:calc_dist(l,None)) #add the distance to said cluster
#     combos_w_euc_dist_grouped = combos_w_euc_dist.groupByKey()
#     closest = combos_w_euc_dist_grouped.mapValues(lambda val: min(list(val))) # filter such that only the closest (dist,clust) remains as value for a vector
#     costs_max += [sum(closest.map(lambda l:l[1][0]).collect())] #add costs for this clustering
#     curr_clusters = closest.map(lambda l : (l[1][1],l[0])).groupByKey() #sort all vectors to its cluster
#     new_clusters = curr_clusters.map(lambda l: calc_new_cluster(l[1])) #calculate new clusters
#     curr_clusters = new_clusters.collect()
#     print('Current cost is', costs_max[-1])


# In[ ]:


plt.plot(costs_rand, label = 'Random')
plt.plot(costs_max, label= 'Max Dist')
plt.legend()
plt.ylabel('Cost')
plt.xlabel('Iteration No.')
plt.title('Cost of k-mean algo using euclidean distance')


# In[ ]:


costs_rand_man = get_costs(cluster_rand,1)
costs_max_man = get_costs(cluster_max,1)


# In[ ]:


plt.plot(costs_rand_man, label = 'Random')
plt.plot(costs_max_man, label= 'Max Dist')
plt.legend()
plt.ylabel('Cost')
plt.xlabel('Iteration No.')
plt.title('Cost of k-mean algo using manhattan distance')


# In[ ]:


a = sc.parallelize([("a",(0,(2,2))), ("b",(0,(3,6))), ("c",(3,(5,2))), ("d",(4,(3,1))),("d",(2,(3,9))),("a",(1,(3,3)))]).groupByKey()
def test(val):
    return min(val)
maxKey = a.mapValues(lambda val: min(list(val)))
maxKey.collect()


# In[ ]:


a = (3,2)
b = (3,1)
sum(np.array([a,b]))/len([a,b])

