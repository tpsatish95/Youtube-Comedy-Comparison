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


DM_Title_Desc = load_obj("PKL/DesignMatrix_Title_Desc")


## Format
## (Title, Desc, Funny/Not)
## for now consider only Tile and Desc OK.
## funny not funny wrong
print(DM_Title_Desc[1])