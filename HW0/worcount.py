import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile(sys.argv[1])


# this reads a line and gets rid of everything that starts with a-zA-Z
def read_line(l):
    sent = re.split(r'[^\w]+', l)
    sent = [w for w in sent if len(w) > 0 and w[0].isalpha()]
    return sent


# flatten
words = lines.flatMap(read_line)
# get rid of stuff


pairs = words.map(lambda w: (w.lower()[0], 1))

# this is for counts
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)
counts.saveAsTextFile(sys.argv[2])

sc.stop()