__author__ = 'satish'

import csv
from collections import defaultdict
import pickle
from xml.dom import minidom

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


ftest = open('comedy_comparisons.test')

unique = defaultdict(int)

csv_fte = csv.reader(ftest)

for row in csv_fte:
	if unique[row[0]] == 0 : # New Video ID
		unique[row[0]] = {'up': 0, 'down': 0}
	if unique[row[1]] == 0 : # New Video ID
		unique[row[1]] = {'up': 0, 'down': 0}
	if row[2] == "left":
		unique[row[0]]["up"] += 1
		unique[row[1]]["down"] += 1
	if row[2] == "right":
		unique[row[0]]["down"] += 1
		unique[row[1]]["up"] += 1

print(len(unique))

save_obj(unique,"UniqueVidUpDown")

path = "../"

DesignMatrix = dict()

title = ""
mediaDescription = ""
comment = []

for ID in unique.keys():
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
