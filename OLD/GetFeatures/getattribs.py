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
(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),
countHint,durationSeconds)

'''

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

########## Entire Design Matrix (Both Test and Train)

# f= load_obj("FunVID")
# nf= load_obj("NotFunVID")

# DesignMatrix = []

# for ID in f:
#     try:
#         doc = minidom.parse("../Data/"+ID+".txt")
#         #doc = minidom.parse("Sample.txt")
#         title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
#         countHint = doc.getElementsByTagName("gd:feedLink")[0].getAttribute("countHint")
#         mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
#         durationSeconds = doc.getElementsByTagName("yt:duration")[0].getAttribute("seconds")
#         ratingAverage = doc.getElementsByTagName("gd:rating")[0].getAttribute("average")
#         ratingMax = doc.getElementsByTagName("gd:rating")[0].getAttribute("max")
#         ratingnNumRaters = doc.getElementsByTagName("gd:rating")[0].getAttribute("numRaters")
#         statisticsViewCount = doc.getElementsByTagName("yt:statistics")[0].getAttribute("viewCount")
#         ratingNumDislikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numDislikes")
#         ratingNumLikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numLikes")

#         featureTuple = (title,mediaDescription,(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),countHint,durationSeconds,1)
#         DesignMatrix.append(featureTuple)       
#         print(featureTuple)
#         print("SS")
#     except:
#         print ('skipped')
#         continue
        
# for ID in nf:
#     try:
#         doc = minidom.parse("../Data/"+ID+".txt")
#         #doc = minidom.parse("Sample.txt")
#         title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
#         countHint = doc.getElementsByTagName("gd:feedLink")[0].getAttribute("countHint")
#         mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
#         durationSeconds = doc.getElementsByTagName("yt:duration")[0].getAttribute("seconds")
#         ratingAverage = doc.getElementsByTagName("gd:rating")[0].getAttribute("average")
#         ratingMax = doc.getElementsByTagName("gd:rating")[0].getAttribute("max")
#         ratingnNumRaters = doc.getElementsByTagName("gd:rating")[0].getAttribute("numRaters")
#         statisticsViewCount = doc.getElementsByTagName("yt:statistics")[0].getAttribute("viewCount")
#         ratingNumDislikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numDislikes")
#         ratingNumLikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numLikes")

#         featureTuple = (title,mediaDescription,(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),countHint,durationSeconds,0)
#         DesignMatrix.append(featureTuple)       
#         print(featureTuple)
#         print("SS")
#     except:
#         print ('skipped')
#         continue

# save_obj(DesignMatrix,"DesignMatrix")

# print(str(DesignMatrix[1])+"\n"+str(DesignMatrix[6000]))

# raw_input("Continue!?")


DoneId = load_obj("finalVidID")
print(len(DoneId))
skip = 0
point = 0
####### Only For Train Data (Design Matrix)

# f= load_obj("uniqueFTrain")
# nf= load_obj("uniqueNFTrain")

DesignMatrix = dict()

for ID in DoneId:
    try:
        doc = minidom.parse("../Data/"+ID.strip()+".txt")
        #doc = minidom.parse("Sample.txt")
        title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
        #countHint = doc.getElementsByTagName("gd:feedLink")[0].getAttribute("countHint")
        mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
        # durationSeconds = doc.getElementsByTagName("yt:duration")[0].getAttribute("seconds")
        # ratingAverage = doc.getElementsByTagName("gd:rating")[0].getAttribute("average")
        # ratingMax = doc.getElementsByTagName("gd:rating")[0].getAttribute("max")
        # ratingnNumRaters = doc.getElementsByTagName("gd:rating")[0].getAttribute("numRaters")
        # statisticsViewCount = doc.getElementsByTagName("yt:statistics")[0].getAttribute("viewCount")
        # ratingNumDislikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numDislikes")
        # ratingNumLikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numLikes")

        #featureTuple = (title,mediaDescription,(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),countHint,durationSeconds,vid[ID])
        #featureTuple = ({ID:[title,mediaDescription]})
        #featureTuple = (title,vid[ID])
        DesignMatrix[ID] = [title,mediaDescription]       
        point+=1
        #print(featureTuple)
        print("SS")
    except:
        print ('skipped')

        skip+=1
        continue
        
# for ID in nf.keys():
#     try:
#         doc = minidom.parse("../Data/"+ID+".txt")
#         #doc = minidom.parse("Sample.txt")
#         title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
#         countHint = doc.getElementsByTagName("gd:feedLink")[0].getAttribute("countHint")
#         mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
#         durationSeconds = doc.getElementsByTagName("yt:duration")[0].getAttribute("seconds")
#         ratingAverage = doc.getElementsByTagName("gd:rating")[0].getAttribute("average")
#         ratingMax = doc.getElementsByTagName("gd:rating")[0].getAttribute("max")
#         ratingnNumRaters = doc.getElementsByTagName("gd:rating")[0].getAttribute("numRaters")
#         statisticsViewCount = doc.getElementsByTagName("yt:statistics")[0].getAttribute("viewCount")
#         ratingNumDislikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numDislikes")
#         ratingNumLikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numLikes")

#         featureTuple = (title,mediaDescription,(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),countHint,durationSeconds,vid[ID])
#         DesignMatrix.append(featureTuple)       
#         print(featureTuple)
#         print("SS")
#     except:
#         print ('skipped')
#         skip+=1
#         continue

# #save_obj(DesignMatrix,"DesignMatrixTrain")

# ####### Only For Test Data (Design Matrix)

# f= load_obj("uniqueFTest")
# nf= load_obj("uniqueNFTest")

# #DesignMatrix = []

# for ID in f.keys():
#     try:
#         doc = minidom.parse("../Data/"+ID+".txt")
#         #doc = minidom.parse("Sample.txt")
#         title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
#         countHint = doc.getElementsByTagName("gd:feedLink")[0].getAttribute("countHint")
#         mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
#         durationSeconds = doc.getElementsByTagName("yt:duration")[0].getAttribute("seconds")
#         ratingAverage = doc.getElementsByTagName("gd:rating")[0].getAttribute("average")
#         ratingMax = doc.getElementsByTagName("gd:rating")[0].getAttribute("max")
#         ratingnNumRaters = doc.getElementsByTagName("gd:rating")[0].getAttribute("numRaters")
#         statisticsViewCount = doc.getElementsByTagName("yt:statistics")[0].getAttribute("viewCount")
#         ratingNumDislikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numDislikes")
#         ratingNumLikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numLikes")

#         featureTuple = (title,mediaDescription,(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),countHint,durationSeconds,vid[ID])
#         DesignMatrix.append(featureTuple)       
#         print(featureTuple)
#         print("SS")
#     except:
#         print ('skipped')
#         skip+=1
#         continue
        
# for ID in nf.keys():
#     try:
#         doc = minidom.parse("../Data/"+ID+".txt")
#         #doc = minidom.parse("Sample.txt")
#         title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
#         countHint = doc.getElementsByTagName("gd:feedLink")[0].getAttribute("countHint")
#         mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
#         durationSeconds = doc.getElementsByTagName("yt:duration")[0].getAttribute("seconds")
#         ratingAverage = doc.getElementsByTagName("gd:rating")[0].getAttribute("average")
#         ratingMax = doc.getElementsByTagName("gd:rating")[0].getAttribute("max")
#         ratingnNumRaters = doc.getElementsByTagName("gd:rating")[0].getAttribute("numRaters")
#         statisticsViewCount = doc.getElementsByTagName("yt:statistics")[0].getAttribute("viewCount")
#         ratingNumDislikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numDislikes")
#         ratingNumLikes = doc.getElementsByTagName("yt:rating")[0].getAttribute("numLikes")

#         featureTuple = (title,mediaDescription,(ratingAverage,ratingMax,ratingnNumRaters),(statisticsViewCount,ratingNumDislikes,ratingNumLikes),countHint,durationSeconds,vid[ID])
#         DesignMatrix.append(featureTuple)       
#         print(featureTuple)
#         print("SS")
#     except:
#         print ('skipped')
#         skip+=1
#         continue

save_obj(DesignMatrix,"DesignMatrix_Title_Desc")

print(str(skip))
print(str(point))
print(len(DesignMatrix))
print("Success!!")
c = input("Continue!?")