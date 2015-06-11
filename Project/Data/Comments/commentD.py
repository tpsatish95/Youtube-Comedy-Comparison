import urllib.request
import urllib
import xml
from xml.dom import minidom
import pickle
from os import listdir


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

# f= load_obj("FunVID")
# nf= load_obj("NotFunVID")

# D = f+nf
D = load_obj("uniqueVidID")

files = listdir(".")

print(len(files))

DoneID = []

for f in files:
    if ".txt" in f and "Sample" not in f:
        DoneID.append(f.replace(".txt",""))

# Done = load_obj("C:/Users/$@T!$#/Downloads/Official/Projects/ML Project/comedy_comparisons/Data/Done/DoneID")
#print(D[1])
# Not Done
NotD = [x for x in D if x not in DoneID]

print(len(D))
#print(len(Done))
print(str(len(NotD)))

SuccessID = []
NSuccessID = []

for ID in NotD:
    u = "https://gdata.youtube.com/feeds/api/videos/"+ID.strip()+"/comments?v=2"
    print (u)
    a=0
    while(a<3):
        try:
            url = urllib.request.urlretrieve(u)
            if url[0] == "":
                continue
            response=open(url[0],"r").read()
            f= open(ID+".txt","w")
            f.write(response)
            f.flush()
            f.close()
            print(ID)
            SuccessID.append(ID)
            break
        except:
            print ('retrying')
            a=a+1
            if a==3:
                NSuccessID.append(ID)
                print("ss")

print("Success!!")
print(len(SuccessID))
print(len(NSuccessID))

save_obj(SuccessID,"SuccessIDC")
save_obj(NSuccessID,"NSuccessIDC")
c = input("Continue!?")