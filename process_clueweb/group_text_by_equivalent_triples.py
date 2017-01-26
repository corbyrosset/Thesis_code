'''
this file groups clueweb textual mentions of triples based on if the triples they represent match, that is p1 == p2 iff ((e_l1, r1, el2) ~ p1 == (e_l2, r2, e_l2) ~ p2) where phrase p1 is an instance of (e_l1, r1, el2). 

The purpose here is investigate whether phrases that are instances of the same
triple can be viewed as paraphrases of each other. Since there are millions of
triples, just use the top K most common, perhaps involving the most common
relationships.

'''
from __future__ import division
import sys
import cPickle as pickle
import operator
import ast
import numpy as np


###############################################################################
###                         	Globals                                     ###
###############################################################################

data_path = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/'
execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/'
manifest_file = 'manifest.txt'
most_common_rels = 'relation_freq.txt' # under data_path
most_common_enty = 'entity_plaintext_freq.txt' # under data_path
num_triples = 5000 ### put top num_triples into the triples_to_extract below
min_words_per_text = 2 ### min number of words needed to count a textual triple
triples_to_extract = set([]) ### the set of triples which are common enough to extract the text instantions of, hopefully there are many text instantiations for at least some of these triples. 

textual_triples = {}

###############################################################################
###                         Helper Methods                                  ###
###############################################################################


def process(file_chunk, filename):

	for i, line in enumerate(file_chunk):
		### [relationships, reversd, (leftEntity, leftEntityID), result, (rightEntity, rightEntityID)]
		line = ast.literal_eval(line) ### convert text to data structure
		text_str = line[3].strip()
		text = text_str.split()
		if len(text) < min_words_per_text:
			continue

		reversd = int(line[1])
		relationships = line[0]
		left = line[2]
		right = line[4]

		if reversd == 1:
			triple = (right, tuple(relationships), left)
		else:
			triple = (left, tuple(relationships), right)

		if triple in triples_to_extract:
			if triple not in textual_triples:
				textual_triples[triple] = set((text_str,))
			else:
				textual_triples[triple].add((text_str,))



###############################################################################
###                                 Main                                    ###
###############################################################################


### open manifest of formatted data
try:
	manifest_list = open(data_path + manifest_file, 'r')
	print 'loaded manifest of which data files need to be processed...'
except:
	print 'no manifest file, aborting'
	exit(1)

### open most common triples file:
try:
	triple_count = open(data_path + 'triple_count.pkl', 'r')
	triple_count = pickle.load(triple_count)
	print 'loaded triple count'
except:
	print 'no pickle file of triple counts, aborting'
	exit(1)
sorted_triples = sorted(triple_count.items(), key=operator.itemgetter(1), reverse=True)
print 'most and least popular triple counts in the top ' + str(num_triples) + \
	' with at least ' + str(min_words_per_text) + ' words per instance:\n' + \
	str(sorted_triples[0][1]) + ' and ' + str(sorted_triples[num_triples-1][1])

triples_to_extract = set(map(lambda x: x[0], sorted_triples[0:num_triples]))

### collect statistics of the corpus chunk by chunk
for chunk in manifest_list:
	chunk = chunk.strip()
	chunk = chunk.replace('processed', 'final')
	try:
		formatted_data_chunk = open(data_path + chunk, 'r')
		print 'processing ' + str(chunk)
	except:
		print 'no \".final\" file exists, run compute_stats_and_structures.py on the .processed files'
		continue
	process(formatted_data_chunk, chunk)


	### 
	'''
	TODO - take most common 400 relationships and most common 100k entities or 
	so and compute a list of triples whose constituents are from these most
	frequent elements. Make sure that these triples have multiple textual
	instantiations (take top K triples ordered by how many textual 
	instantiations there are per triple), and group the textual instantiations 
	mentions by triple to construct paraphrastic equivalence groups. output a 
	list of lists: the outer list is the triple, the inner list is the textual 
	instantiations of those triples.
	'''


	# process(formatted_data_chunk, chunk)

###############################################################################
###                              Persist                                    ###
###############################################################################


### summarize all results
print 'done computing...outputting files...'
print 'data structure sizes in bytes:\n'
print 'textual_triples: ' + str(sys.getsizeof(textual_triples))

textual_triples_sorted = sorted(textual_triples, key=lambda x: len(textual_triples[x]), reverse=True)

report = open(data_path + 'grouped_textual_triples.txt', 'w')
for triple in textual_triples_sorted:
	report.write(str(triple[0:3]) + ' num instances: ' + \
		str(len(textual_triples[triple])) + \
		' avg word count per instance: ' + \
		str((1/len(textual_triples[triple])) * sum([len(t[0].split()) for t in textual_triples[triple]])) + '\n')
report.close()

pickle.dump(textual_triples, open(data_path + 'grouped_textual_triples.pkl', 'w'), pickle.HIGHEST_PROTOCOL)
