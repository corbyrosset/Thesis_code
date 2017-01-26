import numpy as np

### this data from Kristina Toutanova (MSR) from her paper "Representating 
### Text for Joint Embedding of Text and Knowledge Bases"

data_path = '/Users/corbinrosset/Desktop/QA_datasets/FB15K-237_plus_text/Release/'
train = open(data_path + 'train.txt')

counter = 0
most_multiedges, pairwithmost = 0, None
entity_pairs = {}
triples = []
for line in train:
	line = line.strip().split()
	triples.append((line[0], line[1], line[2]))
	counter += 1
	key = (line[0], line[2])
	if (key) in entity_pairs:
		entity_pairs[key] += [line[1]]
		if len(entity_pairs[key]) > most_multiedges:
			most_multiedges = len(entity_pairs[key]) 
			pairwithmost = key
	else:
		entity_pairs[key] = [line[1]]

print 'done parsing ' + str(counter) + ' triples from FB15K-237 dataset'
print most_multiedges
print pairwithmost
print entity_pairs[pairwithmost]