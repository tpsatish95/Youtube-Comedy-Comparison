import pickle

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

TitleandDesc = load_obj("DesignMatrix_Title_Desc")
Comments = load_obj("DesignMatrix_Comments")
IDs = load_obj("finalVidID")

FeatureDict = dict()

for ID in IDs:
	try:
		TD = TitleandDesc[ID]
		C = Comments[ID]
		FeatureDict[ID] = [TD[0],TD[1],C]
		print("S")
	except:
		print("Skipped")
		continue

save_obj(FeatureDict,"FeatureDictTDC")
print(len(FeatureDict))