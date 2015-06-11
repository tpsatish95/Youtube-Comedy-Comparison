import urllib.request
import urllib
import xml
from xml.dom import minidom
import pickle

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

DMTest = load_obj("DesignMatrix")

RevDMTest = []

for item in DMTest:
	moditem = (item[0],item[1],int(float(item[2][0])*int(item[2][2])),item[3][0],int(item[3][2])-int(item[3][1]),item[4],item[5],item[6])
	RevDMTest.append(moditem)

print(RevDMTest[1])

save_obj(RevDMTest,"RevDMTrain")