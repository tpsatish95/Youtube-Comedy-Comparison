__author__ = 'satish'

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
(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),
countHint,durationSeconds)


AND Comments from Different location
'''

path = "./"

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


RankedIDs =  load_obj("rankedVidIds")

DesignMatrix = dict()

title = ""
mediaDescription = ""
comment = []

for ID in RankedIDs:
    try:
        doc = minidom.parse(path + "Data/Meta/"+ID.strip()+".txt")
        # Title from Meta
        title = doc.getElementsByTagName("title")[0].firstChild.nodeValue

        # Description
        try:
            mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
        except:
            mediaDescription = "NONE"
    except:
        print ('No Title :(')
        print("Trying Comments ! ")
        title = "NONE"
        mediaDescription = "NONE"

    try:
        com = minidom.parse(path + "Data/Comments/"+ID.strip()+".txt")
        # Comments
        comment = [c.firstChild.nodeValue for c in com.getElementsByTagName("content")]
    except:
        print("No Comments :(")
        if title == "NONE" and mediaDescription == "NONE":
            print("Nothing :O, SKIP")
            print()
            continue
        comment = []

    DesignMatrix[ID] = [title,mediaDescription,comment]
    print("Got !")
    print()

print(len(DesignMatrix))
save_obj(DesignMatrix,"DesignMatrix")