__author__ = 'satish'

import pickle
from collections import defaultdict
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


test = load_obj("./SandBox/icomedy_comparisons.test")

pairs = defaultdict(int)

# Normalize the test data
print("Before "+ str(len(test)))
for row in test:
	if pairs[row[0]+"|"+row[1]] == 0: # new Video
		pairs[row[0]+"|"+row[1]] = {'left': 0, 'right': 0}
		pairs[row[0]+"|"+row[1]][row[2]] += 1
	else:
		pairs[row[0]+"|"+row[1]][row[2]] += 1
print(len(pairs))

uPairs = defaultdict(int)

keys = list(pairs.keys())

for pair in keys:
	try:
		v1,v2 = pair.split("|")
		if uPairs[v1+"|"+v2] == 0:
			uPairs[v1+"|"+v2] = {'left': 0, 'right': 0}
			uPairs[v1+"|"+v2]["left"] = pairs[v1+"|"+v2]["left"] + pairs[v2+"|"+v1]["left"]
			uPairs[v1+"|"+v2]["right"] = pairs[v1+"|"+v2]["right"] + pairs[v2+"|"+v1]["right"]
		else:
			uPairs[v1+"|"+v2]["left"] = pairs[v1+"|"+v2]["left"] + pairs[v2+"|"+v1]["left"]
			uPairs[v1+"|"+v2]["right"] = pairs[v1+"|"+v2]["right"] + pairs[v2+"|"+v1]["right"]
		pairs[v1+"|"+v2] = {'left': 0, 'right': 0}
		pairs[v2+"|"+v1] = {'left': 0, 'right': 0}
	except:
		continue


finDict = dict()
for k in uPairs.keys():
	if uPairs[k]["left"] != 0 and uPairs[k]["right"] != 0:
		finDict[k] = {'left': uPairs[k]["left"], 'right': uPairs[k]["right"]}
print(len(finDict))

for k in finDict.keys():
	if finDict[k]["left"] > finDict[k]["right"]:
		finDict[k] = "left"
	else:
		finDict[k] = "right"
finalTest = []

for k in finDict.keys():
	finalTest.append([k.split("|")[0],k.split("|")[1],finDict[k]])

save_obj(finalTest,"comedy_comparisons.testPairs")
