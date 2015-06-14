from __future__ import print_function
__author__ = 'satish'

# Adopted From: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#               Olivier Grisel <olivier.grisel@ensta.org>
#               Mathieu Blondel <mathieu@mblondel.org>
#               Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

import numpy as np
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
from sklearn import metrics
import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = "./Models/"

###############################################################################
# Benchmark classifiers
# Bench Mark Result Print Function
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    # print(clf.__name__)

    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    print("classification report:")
    print(metrics.classification_report(y_test, pred,target_names=categories))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    return clf


class Model(object):
	def __init__(self,path):
		self.SentiModel = load_obj(path+"classifier")
		self.ch2 = load_obj(path+"ch2Model")
		self.vectorizer = load_obj(path+"vectorizer")

	def getFNF(self,message):

		vec = self.vectorizer.transform([message])
		Tvec = self.ch2.transform(vec)
		pred = self.SentiModel.predict(Tvec)
		return pred[0]


# funVidID = load_obj("funnyVidID")
# notfunVidID = load_obj("notFunnyVidID")

# # map dict has all classes
# DataDict = load_obj("ProcessedDesignMatrix")

# TitleMatrix = []
# DescriptionMatrix = []
# CommentMatrix = []

# T = Model("./Models/Title/")
# D = Model("./Models/Desc/")
# C = Model("./Models/Comm/")

# print("Models Loaded")

# X = []
# y = []

# j=0

# for vidid in [notfunVidID,funVidID]:
#     for k in vidid:
#         TDC = DataDict[k]
#         rank = j
#         TP = 0.5
#         DP = 0.5
#         CP = 0.5
#         if TDC[0] != "none":
#             TP = T.getFNF(TDC[0].encode("utf-8"))
#         if TDC[1] != "none":
#             DP = D.getFNF(TDC[1].encode("utf-8"))
#         COUNT = 0
#         if TDC[2] != []:
#             for com in TDC[2]:
#                 COUNT += C.getFNF(com.encode("utf-8"))
#             CP = COUNT
#         X.append([TP,DP,CP])
#         y.append(rank)
#     print(str(j) + " Data loaded")
#     j = 1
# X_train = np.array(X)
# y_train = np.array(y)

# save_obj(X_train,"BaggedX")
# save_obj(y_train,"Baggedy")

X = load_obj("BaggedX")
y = load_obj("Baggedy")

print('Vecs loaded')

categories = ["NotFunny","Funny"]

skf = cross_validation.StratifiedKFold(y, n_folds=2,shuffle=True)
print(skf)

for train_index, test_index in skf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('data loaded')

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

    print("%d documents - %0.3fMB (training set)" % (
        len(X_train), data_train_size_mb))
    print("%d categories" % len(categories))
    print()

    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()


    print('=' * 80)
    print("L2 penalty")
    # Train Liblinear model
    clf = benchmark(LinearSVC(loss='l2', penalty="l2",dual=False, tol=1e-3))
