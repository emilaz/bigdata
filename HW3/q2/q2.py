#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from pyspark import SparkConf, SparkContext
import re
import itertools

#set context
conf = SparkConf()
sc = SparkContext(conf=conf)


def create_graph(line):
    both = re.split('\t', line)
    return (int(both[0]),int(both[1]))
    
def init_r(n):
    r = np.ones(n)/n
    return r

def init_h(n):
    return np.ones(n)

def update_a(L_trans,h):
    a= np.zeros(len(h))
    for idx in range(len(h)): #fill the vector
        a[idx] = L_trans.lookup(idx+1)[0]@h
    a = a/max(a) #normalize
    return a

def update_h(L,a):
    h = np.zeros(len(a))
    for idx in range(len(a)): #fill the vector
        h[idx] = L.lookup(idx+1)[0]@a
    h = h/max(h) #normalize
    return h

def update_r(M,r,n,beta):
    r_new=np.ones(n)*(1-beta)/n
    for idx in range(len(r_new)):
        r_new[idx]+=M.lookup(idx+1)[0]@r*beta
    return r_new
    
def get_degree(key,val):
    deg = len(val) #how long is the iterable?
    return [(v,(key, 1./deg)) for v in val]

def create_M_row(end, start_deg,n):
    m = np.zeros(n)
    for node_deg in start_deg:
        m[node_deg[0]-1]=node_deg[1]
    return (end,m)

def create_L_row(start, ends,n): #can also be applied to calc l_trans
    l = np.zeros(n)
    indices = np.array([end for end in ends])-1 
    l[indices] = 1
    return (start,l)

def top_5(vec):
    print('Top 5')
    top = (-vec).argsort()[:5]
    for t in top:
        print('Index = {}, Score = {}'.format(t+1,vec[t]))

def bottom_5(vec):
    print('Bottom 5')
    bot = vec.argsort()[:5]
    for t in bot:
        print('Index = {}, Score = {}'.format(t+1,vec[t]))

#file = sc.textFile('./data/graph-small.txt')
file = sc.textFile('./data/graph-full.txt')
edges = file.map(create_graph)
n = max(edges.max()[0],edges.max(lambda l:l[1])[1])
r_0 = init_r(n)
edges_dupl_free = edges.map(lambda l : (l,1)).reduceByKey(lambda n1,n2: n1+n2).map(lambda l: l[0]) #this eliminates duplicate entries
edges_grouped_by_source = edges_dupl_free.groupByKey()
edges_grouped_by_end = edges_dupl_free.groupBy(lambda l: l[1])
end_start_deg = edges_grouped_by_source.flatMap(lambda l : get_degree(l[0],l[1])) #this gives end-start plus degree of start node
M = end_start_deg.groupByKey().map(lambda l: create_M_row(l[0],l[1],n))

r = r_0
for i in range(40):
    r = update_r(M,r,n,.8)


top = top_5(r)
bot = bottom_5(r)

L = edges_grouped_by_source.map(lambda l : create_L_row(l[0],l[1],n))
L_trans = edges_grouped_by_end.map(lambda l: create_L_row(l[0],l[1],n))
h_0 = init_h(n)

h = h_0
for i in tqdm(range(40)):
    a = update_a(L_trans,h)
    h = update_h(L,a)

top = top_5(h)
bot = bottom_5(h)

top = top_5(a)
bot = bottom_5(a)

