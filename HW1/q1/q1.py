import re
import sys
from pyspark import SparkConf, SparkContext
import numpy as np
import itertools

# set context
conf = SparkConf()
sc = SparkContext(conf=conf)
# read file in
lines = sc.textFile('./data/soc-LiveJournal1Adj.txt')


def split(l):
    line = re.split(r'\t', l)
    val = [int(s) for s in line[1].split(',') if s is not '']
    key = int(line[0])
    return (key, val)


def binner(l):
    map_len = max([v[0] for v in l[1]]) + 1
    has = np.zeros(map_len)
    for v in l[1]:
        has[v[0]] = v[1]
    return (l[0], has)


def top_10(kv):
    fr = kv[1]
    if len(fr) == 0:
        return (kv[0],)
    mut_friends = len(fr[fr != 0])
    lim = min(mut_friends, 10)
    indices = (-kv[1]).argsort(kind='mergesort')[:lim]
    return (kv[0], indices)


friends = lines.map(split)

###TRY WITH THE FRIENDS THING
friends = lines.map(split)
# next, get ID's. We need them to ensure that we have recommendations for everyone later on
ids = friends.map(lambda l: (l[0], []))
# next, get simple friend pairs:
friend_pairs = friends.flatMap(lambda l: [(l[0], fr) for fr in l[1]])
# next, get pairs of 2nd grade friends, without those that are already friends
fof = friends.flatMap(lambda l: itertools.permutations(l[1], 2)).subtract(friend_pairs)
# add (ID,0) pairs. Needed so we get recommendations for everyone
# count how often pairs appear
fof_count = fof.map(lambda l: (l, 1)).reduceByKey(lambda n1, n2: n1 + n2)
# get pot friends+their counts per ID
fof_per_id = fof_count.map(lambda l: (l[0][0], (l[0][1], l[1]))).groupByKey()
# put in bins
fof_bins = fof_per_id.map(binner)
# get recommendations
recommens = fof_bins.map(top_10).union(ids).groupByKey().mapValues(lambda l: list(l)[0])

for_lookup = [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]
for ID in for_lookup:
    print(ID, list(recommens.lookup(ID)[0]))

# def calc_support(pair, baskets):
#     support=0
#     for basket in baskets:
#         if all([item in basket for item in pair]):
#             support+=1
#     return (pair,support)

# def calc_supp(pair, baskets):
#     support=0
#     for basket in baskets:
#         if pair[0] in basket and pair[1] in basket:
#             support+=1
#     return (pair,support)