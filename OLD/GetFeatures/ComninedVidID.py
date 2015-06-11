import pickle
from os import listdir

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

fileMeta = listdir("./Meta/")
filesCom = listdir("./Comments/")

print(len(fileMeta))
print(len(filesCom))

DoneIDM = []
DoneIDC = []

for f in fileMeta:
	if ".txt" in f and "Sample" not in f:
		DoneIDM.append(f.replace(".txt","").strip())
for f in filesCom:
	if ".txt" in f and "Sample" not in f:
		DoneIDC.append(f.replace(".txt","").strip())

finalID = set()

finalID = set(DoneIDC).intersection(set(DoneIDM))

finalID = list(finalID)

print(len(finalID))
save_obj(finalID,"finalVidID")