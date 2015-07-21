__author__ = 'satish'

import pickle
import csv

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


DM = load_obj("DesignMatrix")

ftest = open('comedy_comparisons.test')

#unique = defaultdict(int)

csv_fte = csv.reader(ftest)

iRow = []

for row in csv_fte:
	try:
		if DM[row[0]]:
			if DM[row[1]]:
				iRow.append(row)
	except:
		continue

save_obj(iRow,"icomedy_comparisons.test")
print(len(iRow))

# NewDM = load_obj("../ProcessedDesignMatrix")
# IDs = list(set(DM.keys()) - set(NewDM.keys()))


# print(len(IDs))
