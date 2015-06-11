import cmath
import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
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

DM = load_obj("DesignMatrix_Title")
# DMTrain = load_obj("DesignMatrixTrain")
# DMTest = load_obj("DesignMatrixTest")

def get_words_in_dataset(dataset):
    all_words = []
    for (words, sentiment) in dataset:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
      features['contains(%s)' % word] = (word in document_words)
    return features


def classify_dataset(data):
    return \
            classifier.classify(extract_features(data))

## Split In to test and train
fun = []
notfun = []
for i in DM:
    if i[1] :
        fun.append(i[0])
    else:
        notfun.append(i[0])
trainFunLen = int(len(fun)*0.80)
trainNFunLen = int(len(notfun)*0.80)

TrainSentences = []
TestSentences = []
for i in range(0,len(fun)):
    if i < trainFunLen:
        TrainSentences.append([fun[i],1])
    else:
        TestSentences.append([fun[i],1])
for i in range(0,len(notfun)):
    if i < trainNFunLen:
        TrainSentences.append([notfun[i],0])
    else:
        TestSentences.append([notfun[i],0])
#print(TrainSentences[1])
# for item in DMTrain:
#     moditem = [item[0],item[6]]
#     TrainSentences.append(moditem)
    

# filter away words that are less than 3 letters to form the training data
dataset = []
for (words, funny) in TrainSentences:
    words_filtered = [e.lower() for e in nltk.word_tokenize(words) if len(e) >= 3]
    dataset.append((words_filtered, funny))


# extract the word features out from the training data
word_features = get_word_features(\
                    get_words_in_dataset(dataset))



# nltk.classify.util.apply_features(feature_func, toks, labeled=None)[source]
# Use the LazyMap class to construct a lazy list-like object that is analogous to map(feature_func, toks). In particular, if labeled=False, then the returned list-like object’s values are equal to:

# [feature_func(tok) for tok in toks]
# If labeled=True, then the returned list-like object’s values are equal to:

# [(feature_func(tok), label) for (tok, label) in toks]
# The primary purpose of this function is to avoid the memory overhead involved in storing all the featuresets for every token in a corpus. Instead, these featuresets are constructed lazily, as-needed. The reduction in memory overhead can be especially significant when the underlying list of tokens is itself lazy (as is the case with many corpus readers).

# Parameters: 
# feature_func – The function that will be applied to each token. It should return a featureset – i.e., a dict mapping feature names to feature values.
# toks – The list of tokens to which feature_func should be applied. If labeled=True, then the list elements will be passed directly to feature_func(). If labeled=False, then the list elements should be tuples (tok,label), and tok will be passed to feature_func().
# labeled – If true, then toks contains labeled tokens – i.e., tuples of the form (tok, label). (Default: auto-detect based on types.)



# get the training set and train the Naive Bayes Classifier
training_set = nltk.classify.util.apply_features(extract_features, dataset)
# refer to saved html page outside !
print("training...")
classifier = NaiveBayesClassifier.train(training_set)


# read in the test tweets and check accuracy
# to add your own test tweets, add them in the respective files

print("Loading Test Data....")


accuracy=0

emo = ["funny","notfunny"]
num =[0,1]

emdict =dict(zip(emo,num))

print("Classification undergoing....")

# TestSentences = []
datasetTest = []
# for item in DMTest: 
#     TestSentences.append([item[0],item[6]])
for (words, funny) in TestSentences:
    words_filtered = [e.lower() for e in nltk.word_tokenize(words) if len(e) >= 3]
    datasetTest.append((words_filtered, funny))

for data in datasetTest:
        result = classify_dataset(data[0])
        if result == 1 and result == data[1]:
            accuracy+=1 
        if result == 0 and result == data[1]:
            accuracy+=1

print("Results.")
print("Accuracy = "+str(float(accuracy/len(datasetTest))*100))