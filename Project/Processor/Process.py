import sys
import PreprocessClass

Pre = PreprocessClass.Preprocess()
# Pre.process("hw")

raw = Pre.load_obj("../../FeatureDictTDC")
done = Pre.load_obj("processedTDCDict")

NotDKeys = raw.keys() - done.keys()
print("Not DOne = "+str(len(NotDKeys)))
#processed = dict()

i=len(done)+1
for k in NotDKeys:
	#processed[k] = [Pre.process(raw[k][0]),Pre.process(raw[k][1]),[Pre.process(p) for p in raw[k][2]]]
	done[k] = [Pre.process(raw[k][0]),Pre.process(raw[k][1]),[Pre.process(p) for p in raw[k][2]]]
	print(i)
	i+=1
	Pre.save_obj(done,"processedTDCDict")


#Test
# k = list(done.keys())

# print(done[k[2]])