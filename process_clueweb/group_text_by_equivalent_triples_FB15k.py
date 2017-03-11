'''
This program takes a lot of memory...

Only look at "clueweb_FB15k_all-<lhs, rhs, rel, sent>.pkl" files
or perhaps the raw versions thereof. 

Investigate overlap of Clueweb textual mentions and FB15k triples
- How many unique textual mentions are associated with each triple
 	- with the most common triples (have the most examples)
 	- then investigate how paraphrastic these unique textual mentions are for 
 	  a given triple (triple is the label - distant supervision)
- which and how many unique text strings are associated with many triples?
	- if there are many, it confounds the assumption that text is 
	  representative of the triples they are mentions of

'''
from __future__ import division
import sys
import cPickle as pickle
import numpy as np
import scipy.sparse as sp
import operator

###############################################################################
###                         	Globals                                     ###
###############################################################################

data_path = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/'
FB15k_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/'
execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/process_clueweb/'
manifest_file = 'manifest.txt'
input_prefix = 'clueweb_FB15k_'
output_prefix = 'grouped_FB15k_clueweb_triples'

datatyp = 'all' # which .pkl files to load
NUM_FB1K_TRIPLES = 47291696
NUM_UNIQUE_SENTENCES = 6194853


entity2idx = None 
idx2entity = None 
num_entities_union_rels = 0 # sum of |Entities| + |Relatiionships| in FB15k
USE_FINAL = False #True
min_words_per_text = 2 ### min number of words needed to count a textual triple
num_triples = 5000 ### put top num_triples into the triples_to_extract below

### sparse matrices to be output, similar to what FB15_parse.py outputs
LHS = None
RHS = None
REL = None
SENT = []
unique_sent_map = {}
idx_2_sent_map = {}

### data structures to be built
text_per_triple_cntr = {} # count total number of textual instances f.e. triple
unique_text_per_triple = {} # set of unique text mentions f.e. triple
triple_per_text = {} # converse of unique_text_per_triple, i.e. the SIZE of the set of triples labeled on each unique textual instance. 
rels_per_text = {} # set of unique relationships per sentence

###############################################################################
###                                 Main                                    ###
###############################################################################

entity2idx = pickle.load(open(FB15k_path + 'FB15k_entity2idx.pkl', 'r'))
idx2entity = pickle.load(open(FB15k_path + 'FB15k_idx2entity.pkl', 'r'))
num_entities_union_rels = np.max(entity2idx.values()) + 1

datatyp = 'all'
LHS = pickle.load(open(data_path + input_prefix + '%s-lhs.pkl' % datatyp, 'r'))
RHS = pickle.load(open(data_path + input_prefix + '%s-rhs.pkl' % datatyp, 'r'))
REL = pickle.load(open(data_path + input_prefix + '%s-rel.pkl' % datatyp, 'r'))
SENT=pickle.load(open(data_path + input_prefix + '%s-sent.pkl' % datatyp, 'r'))
unique_sent_map = pickle.load(open(data_path + input_prefix + '%s-sent2idx.pkl' % datatyp, 'r'))
idx_2_sent_map = pickle.load(open(data_path + input_prefix + '%s-idx2sent.pkl' % datatyp, 'r'))
print 'loaded necessary FB15k_clueweb data'

### they need to be csc's in order to be fast
assert isinstance(LHS, sp.csc_matrix)
assert isinstance(RHS, sp.csc_matrix) 
assert isinstance(REL, sp.csc_matrix) 
assert isinstance(SENT, np.ndarray)

assert NUM_FB1K_TRIPLES == len(SENT) == np.shape(LHS)[1] == np.shape(RHS)[1] \
    == np.shape(REL)[1]
assert len(unique_sent_map.keys()) == NUM_UNIQUE_SENTENCES == len(idx_2_sent_map.keys())

### count the most common triples by how often they have mentions
rows_lhs, cols_lhs, vals_lhs = sp.find(LHS)
rows_rel, cols_rel, vals_rel = sp.find(REL)
rows_rhs, cols_rhs, vals_rhs = sp.find(RHS)

counter_1 = 0
for lhs, rel, rhs, sent in zip(rows_lhs, rows_rel, rows_rhs, SENT):
	triple = (lhs, rel, rhs)
	int(lhs) and int(rel) and int(rhs) # make sure these dont throw errors
	
	### text_per_triple_cntr
	# if triple not in text_per_triple_cntr:
	# 	text_per_triple_cntr[triple] = 1
	# else:
	# 	text_per_triple_cntr[triple] += 1

	### unique_text_per_triple
	# if triple not in unique_text_per_triple:
	# 	unique_text_per_triple[triple] = set([sent])
	# else:
	# 	unique_text_per_triple[triple].add(sent)

	### triple_per_text
	# if sent not in triple_per_text:
	# 	triple_per_text[sent] = set([triple])
	# else:
	# 	triple_per_text[sent].add(triple)

	### rels_per_text
	if sent not in rels_per_text:
		rels_per_text[sent] = set([rel])
	else:
		rels_per_text[sent].add(rel)

	
	# print idx2entity[lhs], idx2entity[rel], idx2entity[rhs], idx_2_sent_map[sent]

### make triple_per_text only store the lens of the sets, not the sets:
# for k, v in triple_per_text.items():
# 	triple_per_text[k] = len(v)
# 	assert len(v) > 0

### make rels_per_text only store the lens of the sets, not the sets:
for k, v in rels_per_text.items():
	rels_per_text[k] = list(v)
	assert len(v) > 0
assert len(rels_per_text) == NUM_UNIQUE_SENTENCES

# srtd_triples = sorted(text_per_triple_cntr, key=text_per_triple_cntr.__getitem__, reverse=True)
# srtd_text_per_triple = sorted(triple_per_text.items(), key= lambda (k, v): v, reverse=True)

# assert len(unique_text_per_triple.keys()) == len(text_per_triple_cntr.keys())
# print 'Number of unique triples: ' + str(len(unique_text_per_triple.keys()))
# print 'Top triples: '
# for i in srtd_triples[:10]:
# 	print '\t' + str(idx2entity[i[0]]) + ' ' + str(idx2entity[i[1]]) \
# 		+ ' ' + str(idx2entity[i[2]]) + ' count: ' + str(i[1]) \
# 		+ ' num unique texts: ' + str(len(unique_text_per_triple[i]))

# print 'Textual mentions that have the most associated triples: '
# for i in srtd_text_per_triple[:500]:
# 	print str(i[0]) + ' ' + str(idx_2_sent_map[i[0]]) + ' ' + str(triple_per_text[i[0]])

# print '\nTextual mentions that have the least associated triples: '
# for i in srtd_text_per_triple[-100:]:
# 	print str(i[0]) + ' ' + str(idx_2_sent_map[i[0]]) + ' ' + str(triple_per_text[i[0]])

###############################################################################
###                              Persist                                    ###
###############################################################################

#pickle.dump(text_per_triple_cntr, open(data_path + output_prefix + '-counts.pkl', 'w'), -1)
#pickle.dump(unique_text_per_triple, open(data_path + output_prefix + '-text_sets.pkl', 'w'), -1)
#pickle.dump(triple_per_text, open(data_path + output_prefix + '-triple_per_text_count.pkl', 'w'), -1)
pickle.dump(rels_per_text, open(data_path + output_prefix + '-rels_per_text.pkl', 'w'), -1)

print 'done writing triple components to their respective files'

###############################################################################
###                         		Dead                                    ###
###############################################################################


### open most common triples files, o
# try:
# 	triple_count = open(data_path + 'triple_count.pkl', 'r')
# 	triple_count = pickle.load(triple_count)
# 	print 'loaded triple count'
# except:
# 	print 'no pickle file of triple counts, aborting'
# 	exit(1)
# sorted_triples = sorted(triple_count.items(), key=operator.itemgetter(1), reverse=True)
# print 'most and least popular triple counts in the top ' + str(num_triples) + \
# 	' with at least ' + str(min_words_per_text) + ' words per instance:\n' + \
# 	str(sorted_triples[0][1]) + ' and ' + str(sorted_triples[num_triples-1][1])

# triples_to_extract = set(map(lambda x: x[0], sorted_triples[0:num_triples]))

### summarize all results

