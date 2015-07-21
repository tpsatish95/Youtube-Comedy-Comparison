__author__ = 'satish'

import sys
sys.path.append("./Processor/")
import PreprocessClass

import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Initializing the processor
p = PreprocessClass.Preprocess()

DM = load_obj("DesignMatrix")

i = 0

NewDM = dict()
try:
    NewDM = load_obj("ProcessedDesignMatrix")
    IDs = list(set(DM.keys()) - set(NewDM.keys()))

except:
    NewDM = dict()
    IDs = DM.keys()

for k in IDs:
    NewDM[k] = [p.process(DM[k][0]),p.process(DM[k][1]),[p.process(c) for c in DM[k][2]]]
    i+=1
    if  i % 100 == 0:
        print("Batch " + str(i/100) + " of " + str(len(IDs)/100))
        save_obj(NewDM,"ProcessedDesignMatrix")

save_obj(NewDM,"ProcessedDesignMatrix")
