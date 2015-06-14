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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.utils.extmath import density
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


    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print("top 10 keywords per class:")

        rank = np.argsort(clf.coef_[0])
        top10 = rank[-10:]
        bottom10 = rank[:10]
        print("%s: %s" % ("Funny: ", " ".join(feature_names[top10]).encode("utf-8")))
        print("%s: %s" % ("Not Funny: ", " ".join(feature_names[bottom10]).encode("utf-8")))

        print()

    return clf


funVidID = load_obj("funnyVidID")
notfunVidID = load_obj("notFunnyVidID")

# map dict has all classes
DataDict = load_obj("ProcessedDesignMatrix")

TitleMatrix = []
DescriptionMatrix = []
CommentMatrix = []

j=0

for vidid in [notfunVidID,funVidID]:
    for k in vidid:
        TDC = DataDict[k]
        rank = j
        if TDC[0] != "none":
            TitleMatrix.append([TDC[0].encode("utf-8"),rank])
        if TDC[1] != "none":
            DescriptionMatrix.append([TDC[1].encode("utf-8"),rank])
        if TDC[2] != []:
            for com in TDC[2]:
                CommentMatrix.append([com.encode("utf-8"),rank])
    j=1

# Memory Error Test (FIXED with utf-8 encoding)

# print(len(CommentMatrix))
#
# print(CommentMatrix[1])
#
# print(len(CommentMatrix[1][0]))

# count= 0
#
# for i in CommentMatrix:
#     count += len(i[0])
#
# print(count)

# For Title
X_Title = []
y_Title = []

for i in TitleMatrix:
    X_Title.append(i[0])
    y_Title.append(i[1])

# For Desc
X_Desc = []
y_Desc = []

for i in DescriptionMatrix:
    X_Desc.append(i[0])
    y_Desc.append(i[1])

# For Comments
X_Com = []
y_Com = []

for i in CommentMatrix:
    X_Com.append(i[0])
    y_Com.append(i[1])


Models = ["Title", "Desc", "Comm"]
ModelVecs = [(X_Title,y_Title),(X_Desc,y_Desc),(X_Com,y_Com)]

# Models = [ "Comm"]
# ModelVecs = [(X_Com,y_Com)]

i = 0

for X_,y_ in ModelVecs:

    print('=' * 80)
    print('=' * 80)
    print(Models[i] + " Trainer CrossVal Results")
    print('=' * 80)
    print('=' * 80)

    X_train = np.array(X_)
    y_train = np.array(y_)

    print('data loaded')

    # categories = ["1","2","3","4","5","6","7","8","9","10"]
    # categories = ["1","2","3","4","5"]
    categories = ["1","2"]

    def size_mb(docs):
        return sum(len(s) for s in docs) / 1e6

    data_train_size_mb = size_mb(X_train)

    print("%d documents - %0.3fMB (training set)" % (
        len(X_train), data_train_size_mb))
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

    save_obj(vectorizer,path + Models[i]+"/vectorizer")
    # mapping from integer feature name to original token string
    Rfeature_names = vectorizer.get_feature_names()

    ### Vary K Value
    k=5000
    print("Extracting "+str(k) +" best features by a chi-squared test")
    t0 = time()
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)

    feature_names = [Rfeature_names[i] for i in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
    print()
    feature_names = np.asarray(feature_names)
    save_obj(ch2,path + Models[i]+"/ch2Model")
    results = []

    print('=' * 80)
    print("L2 penalty")
    # Train Liblinear model
    clf = benchmark(LinearSVC(loss='l2', penalty="l2",dual=False, tol=1e-3))

    save_obj(clf,path + Models[i]+"/classifier")

    i += 1