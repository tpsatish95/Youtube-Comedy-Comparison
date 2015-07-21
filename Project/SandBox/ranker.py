__author__ = 'satish'

import pickle
import collections
import operator
import math

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f,  protocol=2)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

# To Chunkify
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


VidIDUpDown = load_obj("UniqueVidUpDown")

for k in VidIDUpDown.keys():
    VidIDUpDown[k]["count"] = VidIDUpDown[k]["up"] + VidIDUpDown[k]["down"]

# print(VidIDUpDown.items())

save_obj(VidIDUpDown,"UniqueVidUpDown")

VidIDUpDownSorted = collections.OrderedDict(sorted(VidIDUpDown.items(), key=lambda x: x[1]['count'], reverse=True))

print(VidIDUpDownSorted.items())

save_obj(VidIDUpDownSorted,"SortedUniqueVidUpDown")

highest = []
lowest = []

for k in VidIDUpDownSorted.items():
    if k[1]["up"] > k[1]["down"]:
        highest.append(k[0])
    else:
        lowest.append(k[0])

Highflat = []
Lowflat = []

for k in highest:
    Highflat.append((k,VidIDUpDownSorted[k]["up"],VidIDUpDownSorted[k]["down"],VidIDUpDownSorted[k]["count"]))
for k in lowest:
    Lowflat.append((k,VidIDUpDownSorted[k]["up"],VidIDUpDownSorted[k]["down"],VidIDUpDownSorted[k]["count"]))

# HighflatSorted = sorted(Highflat, key = operator.itemgetter(3,1,2), reverse=True)
# LowflatSorted = sorted(Lowflat, key = operator.itemgetter(3,2,1), reverse=True)

HighflatSorted = sorted(Highflat, key = lambda t: (t[3],t[1],-t[2]), reverse=True)
LowflatSorted = sorted(Lowflat, key = lambda t: (t[3],t[2],-t[1]), reverse=True)

highest =[]
lowest = []

for hf in HighflatSorted:
    highest.append(hf[0])
for lf in LowflatSorted:
    lowest.append(lf[0])

TempRanks = highest + lowest[::-1]    # add highest and lowests' reverse

print(TempRanks[len(TempRanks)-1])

save_obj(TempRanks,"rankedVidIds")


# Split to 10 classes

Part10 = chunks(TempRanks,math.ceil(len(TempRanks)/10))

save_obj(Part10,"Chunky10Ranked")

# print(len(Part10))

# We dont have data for all VidID so remove those and form new rankings

Ranks = []

DM = load_obj("DesignMatrix")

for r in TempRanks:
    try:
        if DM[r]:
            Ranks.append(r)
    except:
        print("Skipped")
        continue
# Chunk It
Part10i = chunks(Ranks,math.ceil(len(Ranks)/10))
save_obj(Part10i,"Chunky10Rankedi")


######## JUST 2 Classes ( F and NF )

TrueHighest = []
TrueLowest = []

for r in highest:
    try:
        if DM[r]:
            TrueHighest.append(r)
    except:
        print("Skipped")
        continue
for r in lowest:
    try:
        if DM[r]:
            TrueLowest.append(r)
    except:
        print("Skipped")
        continue

save_obj(TrueHighest,"funnyVidID")
save_obj(TrueLowest,"notFunnyVidID")
