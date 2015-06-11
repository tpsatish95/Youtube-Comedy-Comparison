import urllib.request
import urllib
import xml
from xml.dom import minidom
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import cluster, datasets, preprocessing


'''
Feature List:

title
gd:feedLink ---  countHint
media:description
yt:duration --- seconds
gd:rating --- average --- max --- numRaters
yt:statistics (--- favoriteCount) --- viewCount
yt:rating --- numDislikes --- numLikes

Feature Tuple:

(title,mediaDescription,
ratingAverage * ratingnNumRaters,statisticsViewCount*(ratingNumLikes - ratingNumDislikes),
countHint,durationSeconds)

'''

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

DMTest = load_obj("DesignMatrixTrain")

RevDMTest = []
X_Train = []
Y_Train = []

for item in DMTest:
	moditem = (int(float(item[2][0])*int(item[2][2])),int(item[3][0]),int(item[3][2])-int(item[3][1]),int(item[4]),int(item[5]),item[6])
	RevDMTest.append(moditem)
	#ONE X_Train.append([float(float(item[2][0])*int(item[2][2])),float(item[3][0]),float(item[3][2])-float(item[3][1]),float(item[4]),float(item[5])])
	#TWO X_Train.append([float(float(item[2][0])*int(item[2][2])),float(item[3][2])-float(item[3][1]),float(item[4]),float(item[5])])
	#THREE X_Train.append([float(float(item[2][0])*int(item[2][2])),float(item[3][2])-float(item[3][1]),float(item[5])])
	#FOUR X_Train.append([float(item[3][2])-float(item[3][1]),float(item[5])])
	#FIVE 
	X_Train.append([float(item[5])])
	Y_Train.append(item[6])
print("TEst"+str(RevDMTest[1]))


X = np.array(X_Train)
print(X[1])
min_max_scaler = preprocessing.MinMaxScaler()
X_Scaled_Feature_Vecs = min_max_scaler.fit_transform(preprocessing.normalize(X))

print(X_Scaled_Feature_Vecs[5])

C_range = 10.0 ** np.arange(-3, 3)
gamma_range = [0.0]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedKFold(y=Y_Train, n_folds=10)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_Scaled_Feature_Vecs, Y_Train)
store = open("Result.txt","w")
store.write(str(grid.best_estimator_))
store.write("\n"+str(grid.best_score_))
store.close()
save_obj(grid,"TestGrid")
print(str(grid.best_estimator_)+"\n"+str(grid.best_score_))
print("Done.")