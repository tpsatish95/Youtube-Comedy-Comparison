__author__ = 'satish'

# Adopted From: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#               Olivier Grisel <olivier.grisel@ensta.org>
#               Mathieu Blondel <mathieu@mblondel.org>
#               Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import numpy as np
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


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
        print("%s: %s" % ("Funny: ", " ".join(feature_names[top10]).encode("utf-8")))
        print("%s: %s" % ("Not Funny: ", " ".join(feature_names[bottom10]).encode("utf-8")))

        print()

    print("classification report:")
    print(metrics.classification_report(y_test, pred,target_names=categories))

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


# read in 10 classes of [Most Funny,.... ,Least Funny]
Part10 = load_obj("Chunky10Ranked")


map = dict()
i = 1
loop = 1
#concatenation factor of 10parts ;;; if concat=5 => 10/5 = 2Parts
concat = 1

for part in Part10:
    for vid in part:
        map[vid[0]] = i
    if loop%concat == 0:
        i+=1
    loop+=1

# map dict has all classes
DataDict = load_obj("ProcessedDesignMatrix")

TitleMatrix = []
DescriptionMatrix = []
CommentMatrix = []

for k in DataDict.keys():
    TDC = DataDict[k]
    rank = map[k]
    if TDC[0] != "NONE":
        TitleMatrix.append([TDC[0],rank])
    if TDC[1] != "NONE":
        DescriptionMatrix.append([TDC[0],rank])
    if TDC[2] != []:
        for com in TDC[2]:
            CommentMatrix.append([com,rank])

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


Models = ["TITLE", "DESCRIPTION", "COMMENTS"]
ModelVecs = [(X_Title,y_Title),(X_Desc,y_Desc),(X_Com,y_Com)]

i = 0

for X_,y_ in ModelVecs:

    print('=' * 80)
    print('=' * 80)
    print(Models[i] + "Trainer CrossVal Results")
    print('=' * 80)
    print('=' * 80)

    X = np.array(X_)
    y = np.array(y_)

    skf = cross_validation.StratifiedKFold(y, n_folds=10,shuffle=True)
    print(skf)

    for train_index, test_index in skf:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print('data loaded')

        categories = ["1","2","3","4","5","6","7","8","9","10"]
        # categories = ["1","2","3","4","5"]
        # categories = ["1","2"]

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
        print("Extracting "+str(kkk) +" best features by a chi-squared test")
        t0 = time()
        ch2 = SelectKBest(chi2, k=kkk)
        X_train = ch2.fit_transform(X_train, y_train)
        X_test = ch2.transform(X_test)

        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
        print()
        feature_names = np.asarray(feature_names)

        results = []

        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
                (KNeighborsClassifier(n_neighbors=3), "kNN"),
                (RandomForestClassifier(n_estimators=100), "Random forest")
                ):
            print('=' * 80)
            print(name)
            benchmark(clf)

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            benchmark(LinearSVC(loss='l2', penalty=penalty,dual=False, tol=1e-3))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        benchmark(MultinomialNB(alpha=.01))
        benchmark(BernoulliNB(alpha=.01))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        benchmark(Pipeline([
          ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
          ('classification', LinearSVC())
        ]))

        i += 1