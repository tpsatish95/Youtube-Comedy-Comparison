import urllib.request
import urllib
import xml
from xml.dom import minidom
import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


DoneId = load_obj("finalVidID")
print(len(DoneId))
skip = 0
point = 0

DesignMatrix = dict()

for ID in DoneId:
    try:
        doc = minidom.parse("../Data/Comments/"+ID.strip()+".txt")
        #doc = minidom.parse("Sample.txt")
        com = [c.firstChild.nodeValue for c in doc.getElementsByTagName("content")]
        # featureTuple = ({ID:com})
        DesignMatrix[ID] = (com)       
        point+=1
        #print(com)
        print("SS")
    except:
        print ('skipped')
        skip+=1

save_obj(DesignMatrix,"DesignMatrix_Comments")

print(str(skip))
print(str(point))
print(len(DesignMatrix))
print("Success!!")
c = input("Continue!?")