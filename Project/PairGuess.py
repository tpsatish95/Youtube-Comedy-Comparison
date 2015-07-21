__author__ = 'satish'

import numpy as np
from time import time
from sklearn.svm import SVR
from xml.dom import minidom

#### Main Path
p = "./"
import sys
sys.path.append(p + "Processor/")
import PreprocessClass

import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

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

class PairGuess(object):
    def __init__(self):
        self.T = Model(p + "Models/Title/")
        self.D = Model(p + "Models/Desc/")
        self.C = Model(p + "Models/Comm/")
        #Load SVR Classifier Model
        self.svr = load_obj(p + "BaggedSVRClassifierModel")
        # print("Models Loaded")
        self.p = PreprocessClass.Preprocess()

    def getTDC(self,Vid):

        try:
            doc = minidom.parse(p + "Data/Meta/"+Vid.strip()+".txt")
            # Title from Meta
            title = doc.getElementsByTagName("title")[0].firstChild.nodeValue
            # Description
            try:
                mediaDescription = doc.getElementsByTagName("media:description")[0].firstChild.nodeValue
            except:
                mediaDescription = "NONE"
        except:
            title = "NONE"
            mediaDescription = "NONE"
        try:
            com = minidom.parse(p + "Data/Comments/"+Vid.strip()+".txt")
            # Comments
            comment = [c.firstChild.nodeValue for c in com.getElementsByTagName("content")]
        except:
            comment = []

        return [title,mediaDescription,comment]

    def getProcessed(self,TDC):
        return [self.p.process(TDC[0]),self.p.process(TDC[1]),[self.p.process(c) for c in TDC[2]]]

    def getVec(self,TDC):
        TP = 0.5
        DP = 0.5
        CP = 0.5  # To denote no Comments
        if TDC[0] != "none":
            TP = self.T.getFNF(TDC[0].encode("utf-8"))
        if TDC[1] != "none":
            DP = self.D.getFNF(TDC[1].encode("utf-8"))
        COUNT = 0
        if TDC[2] != []:
            for com in TDC[2]:
                COUNT += self.C.getFNF(com.encode("utf-8"))
            CP = COUNT
        return np.array([TP,DP,CP])

    def getScore(self,vec):
        return self.svr.predict(vec)[0]

    def getLR(self,vid1,vid2):
        s1 = self.getScore(self.getVec(self.getProcessed(self.getTDC(vid1))))
        s2 = self.getScore(self.getVec(self.getProcessed(self.getTDC(vid2))))
        if s1>s2:
            return "left"
        # elif s1 == s2:
        #     return "same"
        else:
            return "right"
