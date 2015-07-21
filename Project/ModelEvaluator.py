__author__ = 'satish'

import pickle
import PairGuess
from sklearn import metrics

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

pairTest = load_obj("./SandBox/comedy_comparisons.testPairs")

X = []
y = []
for i in pairTest:
	X.append([i[0],i[1]])
	y.append(i[2])

PG = PairGuess.PairGuess()

pred = []

j=0
for i in X:
	pred.append(PG.getLR(i[0],i[1]))
	print([i,y[j],pred[j],metrics.accuracy_score(y[:j+1], pred)])
	j += 1
score = metrics.accuracy_score(y, pred)
print("accuracy:   %0.3f" % score)

print("classification report:")
categories = ["Left","Right"]
print(metrics.classification_report(y, pred,target_names=categories))

print("confusion matrix:")
print(metrics.confusion_matrix(y, pred))

# Result was 53%
