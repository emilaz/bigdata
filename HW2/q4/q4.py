#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def read_ratings(link):
    f = open(link)
    for idx,l in enumerate(f.readlines()):
        user_ratings = np.array([int(movie) for movie in l.split(' ') if len(movie)>0])
        if idx == 0:
            ratings = user_ratings[None,:]
        else:
            ratings = np.append(ratings,user_ratings[None,:],axis=0)
    return ratings

def compute_p_q(ratings):
    # first, compute the degree matrix over users (P)
    p_entries = np.sum(ratings, axis = 1)
    p = np.diag(p_entries)
    q_entries = np.sum(ratings, axis = 0)
    q = np.diag(q_entries)
    return p,q

def raise_to_power(degree_mat):
    ret = degree_mat**(-.5)
    ret[np.isinf(ret)]=0
    return ret

def compute_gamma_user(p,ratings):
    scale = raise_to_power(p)
    gamma = scale@ratings@ratings.T@scale@ratings
    return gamma

def compute_gamma_item(q,ratings):
    scale = raise_to_power(q)
    gamma = ratings@scale@ratings.T@ratings@scale
    return gamma

def get_highest_sim_scores(recoms,user):
    first_100 = recoms[user,:100] #get set S, which is the first 100 movies
    top_5 = np.argsort(-first_100,kind ='mergesort')[:5]
    return top_5


ratings = read_ratings('./data/user-shows.txt')

ratings.shape

shows= open('./data/shows.txtws.txt').readlines()

p, q = compute_p_q(ratings)

recos_user = compute_gamma_user(p,ratings)

top5_user = get_highest_sim_scores(recos_user,499)

for top in top5_user:
    print(shows[top], recos_user[499,top])

recos_item = compute_gamma_item(q,ratings)
top5_item = get_highest_sim_scores(recos_item,499)
for top in top5_item:
    print(shows[top], recos_item[499,top])

