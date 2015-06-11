import urllib.request
import urllib
import pickle
from os import listdir

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

path = "./data/"

D = load_obj("UniqueVidUpDown")

D = D.keys()

files = listdir(path)

print(len(files))

DoneID = []

for f in files:
    if ".txt" in f:
        DoneID.append(f.replace(".txt",""))

NotD = [x for x in D if x not in DoneID]

print("Total IDs "+str(len(D)))
print("Total Not IDs "+str(len(NotD)))

for ID in NotD:
    u = "https://gdata.youtube.com/feeds/api/videos/"+ID.strip()+"?v=2"
    print (u)
    a=0
    while(a<3):
        try:
            url = urllib.request.urlretrieve(u)
            if url[0] == "":
                continue
            response=open(url[0],"r").read()
            f= open(path+ID+".txt","w")
            f.write(response)
            f.flush()
            f.close()
            print(ID)
            break
        except:
            print ('retrying')
            a=a+1

print("Success!!")

c = input("Continue!?")
