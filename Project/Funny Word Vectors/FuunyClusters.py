'''
Copyright 2015 Satish Palaniappan

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at,

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.,
See the License for the specific language governing permissions and limitations under the License.
'''
__author__ = "Satish Palaniappan"


import numpy as np
from sklearn import cluster, datasets, preprocessing
import pickle
import gensim
from collections import defaultdict
import time
import sys
import pyprind
import logging
logging.basicConfig(level=logging.INFO)
import operator
import Kmeans


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = "./Models/"

model = gensim.models.Word2Vec.load_word2vec_format('vectors.bin', binary=True,norm_only = True)

allseeds = []
words = []
for word in model.vocab:
    allseeds.append(tuple(model[word]))
    words.append(word)

print("Vectors Loaded.")

cat = ['funny']

init_seeds = [tuple(model[cat[0]])]

print("InitSeeds Loaded.")

# init curVecs
curVecs = allseeds

'''
Degree Analyzer (Chooses best granularity)
'''

degree = 5
print(str(degree) + " closest points per cluster")

catSeeds = dict()
for i in range(0,len(cat)):
    catSeeds[cat[i]] = i

Rseeds = init_seeds

t = len(Rseeds)
mv = len(model.vocab)
numInter = 0
while t < mv:
    numInter += 1
    mv -= t
    t = t*degree

#save_obj(init_seeds,path + "init_seeds")
#save_obj(allseeds,path + "allseeds")


threshold = len(Rseeds)
mv = len(model.vocab)
level = 1
while threshold < mv and level <= 10:
    # KMeans
    print()
    print()
    print(str(numInter) + " more iterations left ...")
    print()
    t0 = time.time()

    curVecs = list(set(curVecs) - set(Rseeds))


    centers , xtoc , distances = Kmeans.kmeans(np.array(curVecs), np.array(Rseeds), maxiter=1, metric='cosine')
    print(str(time.time()-t0))

    print("Level "+ str(level) + " Done.")
    kmeans = centers
    # save_obj(kmeans,path+"Level"+str(level))

    print("Mapping Centroids to Labels")
    centroid2Word = dict()
    for c in Rseeds:
        w = words[allseeds.index(tuple(c))]
        centroid2Word[Rseeds.index(c)] = w
    # save_obj(centroid2Word,path+"Level"+str(level)+"c2w")

    print()

    print("Computing the next level of seeds")
    temp = []

    transform = distances

    distPart = sorted(zip(range(0,transform.shape[0]),np.argmax(transform, axis = 1),np.amax(transform,axis = 1)), key = operator.itemgetter(1,2), reverse = True)
    distPartDict = defaultdict(list)
    for i in distPart:
        distPartDict[i[1]].append(i[0])

    transform = [] # Memory
    distances = [] # Memory

    # ProgressBar setup
    my_prbar = pyprind.ProgBar(len(Rseeds))
    for index,cent in enumerate(Rseeds):
        ind = distPartDict[index][:degree]
        newSeeds = np.array(curVecs)[ind]

        temp.extend((tuple(n) for n in newSeeds))

        curSeed = catSeeds[words[allseeds.index(tuple(cent))]]
        for new in newSeeds:
            catSeeds[words[allseeds.index(tuple(new))]] = curSeed

        # update Progressbar
        my_prbar.update()

    # save_obj(catSeeds,path+"seed2Domain")

    # Next Level of seeds (ONLY)
    wordsTemp = [words[allseeds.index(tuple(n))] for n in temp]
    print(wordsTemp)
    save_obj(wordsTemp,path + "words"+str(level))
    Rseeds = temp
    # print(len(temp))

    # Increasing the Threshold
    mv -= threshold
    threshold = threshold*degree
    level += 1
    numInter -= 1
