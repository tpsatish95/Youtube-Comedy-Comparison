import csv
from collections import defaultdict
import pickle


def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)


ftest = open('comedy_comparisons.test')
ftrain = open('comedy_comparisons.train')

count = dict()

csv_fte = csv.reader(ftest)
csv_ftr = csv.reader(ftrain)

c=0

# Zero all 
for row in csv_ftr:
	count[row[0]]=0
	count[row[1]]=0
for row in csv_fte:
	count[row[0]]=0
	count[row[1]]=0

#Rewind
ftest.seek(0)
ftrain.seek(0)

# Ucount used to get the number of times each video has occuured in all training samples
ucount = defaultdict(int)
pairs=defaultdict(int)
#to count in a given pair how many times the left video gets funny rating
pairLeftResults = defaultdict(int)
#get no of times a Pair gets compared, No of items bellow a video 
for row in csv_ftr:
	c+=1
	if row[2] == "left":
		count[row[0]]+=1
	elif row[2] == "right":
		count[row[1]]+=1
	ucount[row[0]] +=1
	ucount[row[1]] +=1
	if pairs[row[1],row[0]] != 0:
		pairs[row[1],row[0]]+=1
		if row[2]=="right":
			pairLeftResults[row[1],row[0]]+=1
	else:
		pairs[row[0],row[1]]+=1
		if row[2]=="left":
			pairLeftResults[row[0],row[1]]+=1
	

for row in csv_fte:
	c+=1
	if row[2] == "left":
		count[row[0]]+=1
	elif row[2] == "right":
		count[row[1]]+=1
	ucount[row[0]] +=1
	ucount[row[1]] +=1
	if pairs[row[1],row[0]] != 0:
		pairs[row[1],row[0]]+=1
		if row[2]=="right":
			pairLeftResults[row[1],row[0]]+=1
	else:
		pairs[row[0],row[1]]+=1
		if row[2]=="left":
			pairLeftResults[row[0],row[1]]+=1



print("Unique count "+str(len(count)))
print("Total instances "+str(c))
print(max(ucount.values()))
print(len(set(ucount.values())))

cou=0
for i in pairs.keys():
	if pairs[i] != 0:
		cou+=1

print("Pairs Check: "+ str(cou))
save_obj(pairLeftResults,"classified pairwise comparisons/pairLeftCountResultsDict")

print("Number of iterations "+str(len(set(pairs.values()))))
li = []
# 0 eliminate since spurious pairs compared
for value in set(pairs.values())-{0}:
	for key in pairs.keys(): 
		if pairs[key] == value:
			li.append(key)
	print(value)
	#categorize the dataset based on number of times each pair gets compared
	save_obj(li,"classified pairwise comparisons/"+str(value)+"Comparisons")
	li=[]

# Ranking Code (Checking...)
# rank = dict()
# r=defaultdict(int)

# for key in count.keys():
# 	rank[count[key]] = key
# 	r[count[key]] +=1 

# print(str(len(rank)))
# print(rank[0] + rank[2] + rank[len(rank)-1])