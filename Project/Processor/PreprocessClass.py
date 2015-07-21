from os import listdir
import pickle
import re, collections
import twokenize
from nltk.corpus import stopwords
#### USAGE   " ".join(tokenize(line))  ## line is the line to be tokenized
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
# USAGE stemmer.stem(String)

path = "./Processor/"

class Preprocess(object):
	
	def __init__(self):
		
		self.acronyms = self.load_obj("acronymsDict")
		self.emoticons = self.load_obj("SmileyDict")
		self.contractions = self.load_obj("contractionsDict")
		self.wordDict = self.load_obj("wordDict")		
		# remove abrreviations
		# find intersection between dict words and acronyms to eliminate only real acronyms
		#k1 = self.wordDict.keys()
		# acronyms =  dict((k, v) for (k, v) in acronyms.items() if k not in k1)
		self.wordDict = dict((k, v) for (k, v) in self.wordDict.items() if k not in self.acronyms)
		self.wordDict = dict((k, v) for (k, v) in self.wordDict.items() if k not in self.contractions)
		#test
		#print(wordDict["lol"]) failed
		#stop = [w for w in stopwords.words('english')] #if len(w)<=3]
		#stanford nlp stop word list
		self.stop  = ["me","he","she","i","a","an","and","are","as","at","be","by","for","from","has","he","is","in","it","its","of","on","that","the","to","was","were","will","with"]
		#print(stop)
		self.stop+=[")","(",".","'",",",";",":","?","/","!","@","$","*","+","-","_","=","&","%","`","~","\"","{","}"]
		#print(stop)

		self.NWORDS = self.train(self.words(open(path + 'big.txt').read()))

		self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
		self.tags = ["joy","anger","sadness","love","fear","surprise"]
		self.remove = ["joy","joyful","anger","sad","sadness","fearful","love","fear","surprise","surprised","surprising","surprisingly"]


	def save_obj(self,obj, name ):
	    with open( name + '.pkl', 'wb') as f:
	        pickle.dump(obj, f,  protocol=2)

	def load_obj(self,name ):
	    with open( path + name + '.pkl', 'rb') as f:
	        return pickle.load(f)




	##############################
	# spell trainer (Peter Norvig)

	def words(self,text): return re.findall('[a-z]+', text.lower()) 

	def train(self,features):
	    model = collections.defaultdict(lambda: 1)
	    for f in features:
	        model[f] += 1
	    return model

	def edits1(self,word):
	   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
	   deletes    = [a + b[1:] for a, b in splits if b]
	   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
	   replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
	   inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
	   return set(deletes + transposes + replaces + inserts)

	def known_edits2(self,word):
	    return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

	def known(self,words): return set(w for w in words if w in self.NWORDS)

	def correct(self,word):
	    candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
	    return max(candidates, key=self.NWORDS.get)

	#test
	#USAGE  correct("word")

	####################################


	def process(self,text):
		
		tTweet = ""
		for word in text.split():
			if "#" in word:
				word = word.replace("#"," ")
				f=0
				for tt in self.remove:
					if tt in word:
						f=1
				if f==1:
					continue
			tTweet = " ".join([tTweet,word])
			tTweet = tTweet.strip()

		tempTweet = ""
		for word in twokenize.tokenize(tTweet):
			if word != " " and word not in self.stop and not word.isdigit():
				word = word.strip().lower()
				if len(word) > 26:
					word=word[:27]
				#### Normalize Emoticons
				try:
					word = self.emoticons[word]
				except:
					#Normalize Acronyms
					try:
						try:
							if  self.wordDict[word] ==1:
								word = word
						except:
							word = self.acronyms[word]
					except:
					#Normalize Contractions
						try:
							word = self.contractions[word]
						except:
							#Normalize words (Spell)
							try:
								if self.wordDict[word] == 1:
									word =	word
							except:
								CW = self.correct(word)
								if "@" in word or "#" in word:
									word = word
								else:
									if CW != "a":
										word = CW
				if "@" in word:
					word="@user"
				tempTweet = " ".join([tempTweet,word.strip()])
				tempTweet = tempTweet.lower().strip()
		tempTweet = " ".join(stemmer.stem(w) for w in tempTweet.split(" ") if w not in self.stop)
		#print(tempTweet.encode("utf-8"))
		return(tempTweet)

##Usage
# pre = Preprocess()
# pre.process("lol god pls help with my hw :) :(:D")
