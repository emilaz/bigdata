import numpy as np
from pyspark import SparkConf, SparkContext
import re
import itertools

#set context
conf = SparkConf()
sc = SparkContext(conf=conf)

def split(l):
    verbs = re.split(' ',l)
    return set([v for v in verbs if len(v)>0])

def calc_support(pair, baskets):
    support=0
    for basket in baskets:
        if all([item in basket for item in pair]):
            support+=1
    return (pair,support)

def calc_conf(kv, prev_supports_dict):
    supp = kv[1]
    cond_1 = prev_supports_dict[kv[0][0]]
    conf_1 = supp/cond_1
    cond_2= prev_supports_dict[kv[0][1]]
    conf_2 = supp/cond_2
    return (kv[0],conf_1),((kv[0][1],kv[0][0]),conf_2)

def find_triples(kv, l2s):
    return list(set([(kv[0],kv[1],l) for pair in l2s for l in pair if (kv[0],l) in l2s and (kv[1],l) in l2s and l not in kv]))

def calc_conf_triples(kv, prev_supports_dict):
    supp = kv[1]
    cond_1 = prev_supports_dict[(kv[0][0],kv[0][1])]
    conf_1 = supp/cond_1
    cond_2= prev_supports_dict[(kv[0][1],kv[0][2])]
    conf_2 = supp/cond_2
    cond_3= prev_supports_dict[(kv[0][0],kv[0][2])]
    conf_3 = supp/cond_3
    return (kv[0],conf_1),((kv[0][1],kv[0][2],kv[0][0]),conf_2),((kv[0][0],kv[0][2],kv[0][1]),conf_3)


s_thresh=100

#read file in
file = sc.textFile('./data/browsing.txt')
all_baskets = file.map(split).collect()
c1 = file.flatMap(split)

#now, count elements
c1_counts = c1.map(lambda l: (l,1)).reduceByKey(lambda n1,n2:n1+n2)

#filter out elements with <100 support, throw away the count for further processing
l1_with_support = c1_counts.filter(lambda l: l[1]>=s_thresh)
l1_with_support_dict = l1_with_support.collectAsMap()
l1 = l1_with_support.map(lambda l: l[0])

# get only keys, do cartesian product on them (make sure to have every pair only once)
c2 = l1.cartesian(l1).filter(lambda a: a[0]<a[1])

#count their occurences in whole dataset:
c2_counts = c2.map(lambda l: calc_support(l,all_baskets))
l2_with_support = c2_counts.filter(lambda l: l[1]>= s_thresh)
# this is needed to calculate c3 later on
l2 = l2_with_support.map(lambda l : l[0])

conf_l2 = l2_with_support.flatMap(lambda l: calc_conf(l,l1_with_support_dict))

# get TOP5, based on confidence
conf_l2.sortBy(lambda l: -l[1]).take(5)

# now, get candidate itemsets of length 3:
l2_with_support_dict = l2_with_support.collectAsMap()

# we need to make sure to drop duplicates with the additional groupBy,map functions
c3 = l2.flatMap(lambda kv: find_triples(kv,l2_with_support_dict.keys())).groupBy(lambda l:l).map(lambda l:l[0])

c3.take(4)

c3_counts = c3.map(lambda l : calc_support(l,all_baskets))

l3_with_support = c3_counts.filter(lambda l: l[1]>=s_thresh)
l3 = l3_with_support.map(lambda l : l[0])

conf_l3 = l3_with_support.flatMap(lambda l : calc_conf_triples(l,l2_with_support_dict))

conf_l3.take(2)

# get TOP5, based on confidence, then based on lexicographical ordering
conf_l3.map(lambda l : (-l[1],l[0][0],l[0][1],l[0][2])).sortBy(lambda l: l).take(5)