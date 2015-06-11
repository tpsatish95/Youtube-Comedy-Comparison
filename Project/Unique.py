__author__ = 'satish'

import csv
from collections import defaultdict
import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


ftrain = open('comedy_comparisons.train')

unique = defaultdict(int)

csv_ftr = csv.reader(ftrain)

for row in csv_ftr:
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
