# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pickle
import chardet


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

# read in 10 classes of funny and not funny datasets
parted10 = load_obj("10PartsVidRanking")

p10 = dict()
i = 1
loop = 1
for part in parted10:
    for vid in part:
        p10[vid[0]] = i
    if loop%5 == 0:
        i+=1
    loop+=1
# p10 dict has all classes

DataDict = load_obj("processedTDCDict")

DesignMatrix = []
for k in DataDict.keys():
    TDC = DataDict[k]
    #DesignMatrix.append([TDC[0] +" "+ TDC[1],p10[k]])   ### only title and desc JOINT
    rank = p10[k]
    # DesignMatrix.append([TDC[0].encode("UTF-8"),rank])
    # DesignMatrix.append([TDC[1].encode("UTF-8"),rank])
    # for com in TDC[2]:
    #     DesignMatrix.append([com.encode("UTF-8"),rank])
    data  = ""
    data  = TDC[0] + " " + TDC[1]
    for com in TDC[2]:
        data = data + " " + com
    DesignMatrix.append([data.encode("UTF-8"),rank])
# print(DesignMatrix[0])
# print(DesignMatrix[1])
# print(DesignMatrix[2])

X=[]
y=[]
for i in DesignMatrix:
    X.append(i[0])
    y.append(i[1])

# print(X[0])
print(len(X))

#decoding
# XD=[]
# for x1 in X:
#     #print(x1)
#     try:
#         XD.append(x1.decode(chardet.detect(x1)['encoding']))
#     except:
#         XD.append(x1)
# print(len(XD))
# XD=[]
# for x in X:
#     XD.append(x.decode("utf-8")) 
# print(len(XD))

X = np.array(X)
y = np.array(y)
skf = cross_validation.StratifiedKFold(y, n_folds=10,shuffle=True)
print(skf)  

for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('data loaded')

    #categories = ["1","2","3","4","5","6","7","8","9","10"]
    #categories = ["1","2","3","4","5"]
    categories = ["1","2"]
    #print(X_train[1])
    def size_mb(docs):
        return sum(len(s) for s in docs) / 1e6

    data_train_size_mb = size_mb(X_train)
    data_test_size_mb = size_mb(X_test)

    print("%d documents - %0.3fMB (training set)" % (
        len(X_train), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(X_test), data_test_size_mb))
    print("%d categories" % len(categories))
    print()

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words=None)
    X_train = vectorizer.fit_transform(X_train)
    duration = time() - t0

    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()

    X_test = vectorizer.transform(X_test)
    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()

    # mapping from integer feature name to original token string
    feature_names = vectorizer.get_feature_names()
    ### Vary K Value
    kkk=3000
    print("Extracting"+str(kkk) +" best features by a chi-squared test")
    t0 = time()
    ch2 = SelectKBest(chi2, k=kkk)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)

    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()
    feature_names = np.asarray(feature_names)


    def trim(s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


    ###############################################################################
    # Benchmark classifiers
    # Bench Mark Result Print Function
    def benchmark(clf):
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        print(clf.__name__)

        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
            print("top 10 keywords per class:")
            
            rank = np.argsort(clf.coef_[0])
            top10 = rank[-10:]
            bottom10 = rank[:10]
            print(trim("%s: %s" % ("Funny: ", " ".join(feature_names[top10]).encode("utf-8"))))
            print(trim("%s: %s" % ("Not Funny: ", " ".join(feature_names[bottom10]).encode("utf-8"))))

            print()

        print("classification report:")
        print(metrics.classification_report(y_test, pred,target_names=categories))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time

    results = []

    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=3), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")
            ):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty=penalty)))

    # Train SGD with Elastic Net penalty
    print('=' * 80)
    print("Elastic-Net penalty")
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet")))

    # Train NearestCentroid without threshold
    print('=' * 80)
    print("NearestCentroid (aka Rocchio classifier)")
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))

    print('=' * 80)
    print("LinearSVC with L1-based feature selection")
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
      ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
      ('classification', LinearSVC())
    ])))

    # # make some plots

    # indices = np.arange(len(results))

    # results = [[x[i] for x in results] for i in range(4)]

    # clf_names, score, training_time, test_time = results
    # training_time = np.array(training_time) / np.max(training_time)
    # test_time = np.array(test_time) / np.max(test_time)

    # plt.figure(figsize=(12, 8))
    # plt.title("Score")
    # plt.barh(indices, score, .2, label="score", color='r')
    # plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    # plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    # plt.yticks(())
    # plt.legend(loc='best')
    # plt.subplots_adjust(left=.25)
    # plt.subplots_adjust(top=.95)
    # plt.subplots_adjust(bottom=.05)

    # for i, c in zip(indices, clf_names):
    #     plt.text(-.3, i, c)

    # plt.show()
