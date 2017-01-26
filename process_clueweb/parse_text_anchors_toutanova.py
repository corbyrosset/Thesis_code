import numpy as np
import pprint

### this data from Kristina Toutanova (MSR) from her paper "Representating 
### Text for Joint Embedding of Text and Knowledge Bases"

'''The textual mentions files have lines like this:

/m/02qkt        [XXX]:<-nn>:fact:<-pobj>:in:<-prep>:game:<-nsubj>:'s:<ccomp>:pivot:<nsubj>:[YYY]    /m/05sb1    3

This indicates the mids of two Freebase entities, together with a fully lexicalized dependency path between the entities. The last element in the tuple is the number of occurrences of the specified entity pair with the given dependency path in sentences from ClueWeb12.
The dependency paths are specified as sequences of words (like the word "fact" above) and labeled dependency links (like <nsubj> above). The direction of traversal of a dependency arc is indicated by whether there is a - sign in front of the arc label "e.g." <-nsubj> vs <nsubj>.
'''

data_path = '/Users/corbinrosset/Desktop/QA_datasets/FB15K-237_plus_text/Release/'
corpus1 = open(data_path + 'text_emnlp.txt')
corpus2 = open(data_path + 'text_cvsc.txt')


counter = 0
textual_triples = []
for line in corpus1:
	# if counter > 1000:
	# 	exit()
	line = line.strip().split()
	line[1] = line[1][5:-5] ### strip [XXX] and [YYY]
	### filters the dependencies
	# line[1] = filter(lambda x: '<' in x or '>' in x, line[1].split(':'))
	line[1] = filter(lambda x: x and '<' not in x and '>' not in x, line[1].strip().split(':'))
	line[1] = ' '.join(line[1])
	if len(line[1]) > 0 and int(line[3]) > 5:
		textual_triples.append((line[0], line[1], line[2], line[3]))
		counter += 1
textual_triples = sorted(textual_triples, key = lambda x: x[3])
pprint.pprint(textual_triples)
print counter
