from __future__ import division
import cPickle as pickle
import numpy as np
import pprint
import ast
import scipy.sparse as sp

data_path = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/'
FB15k_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/'
execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/process_clueweb/'
manifest_file = 'manifest.txt'
output_prefix = 'clueweb_FB15k_'

entity2idx = pickle.load(open(FB15k_path + 'FB15k_entity2idx.pkl', 'r'))
idx2entity = pickle.load(open(FB15k_path + 'FB15k_idx2entity.pkl', 'r'))
num_entities_union_rels = np.max(entity2idx.values()) + 1

### sparse matrices to be output, similar to what FB15_parse.py outputs
LHS = None
RHS = None
REL = None
SENT = []

def process(formatted_data_chunk, chunk):
    '''take a clueweb data chunk (as file descriptor and title), and output 4 matrices: lhs is a sparse array for left entities, just like it is 
        used for FB15k data. rhs, rel are the same, the only difference is 
        sent, which will be a list of strings corresponding to the textual 
        mentions
    '''
    successes = 0
    lhs_id, rhs_id, rel_id = [], [], []
    sentences = []
    for i, line in enumerate(formatted_data_chunk):
        ### format of each line:
        ### [relationships, reversd, (leftEntity, leftEntityID), text, (rightEntity, rightEntityID)]
        line = ast.literal_eval(line) ### convert text to data structure
        reversd = int(line[1])
        relationships = line[0]
        left = line[2]
        text = line[3] 
        right = line[4]
        
        leftID = '/' + left[1].replace('.', '/')
        rightID = '/' + right[1].replace('.', '/')
        relationship = '/' + './'.join([r.replace('.', '/') for r in relationships])
        
        if not (leftID in entity2idx and rightID in entity2idx and relationship in entity2idx):
            continue
        else: # valid FB15k triple
            lhs_id.append(entity2idx[leftID])
            rhs_id.append(entity2idx[rightID])
            rel_id.append(entity2idx[relationship])
            sentences.append(text)
            successes += 1

    ### make sparse matrices of size |enty union rel| by num_examples
    # coo_matrix((data, (row, col)), shape=(4, 4))
    lhs_id, rhs_id, rel_id = np.array(lhs_id), np.array(rhs_id), np.array(rel_id)
    lhs = sp.coo_matrix((np.ones(successes), (lhs_id, np.arange(successes))), \
            shape = (num_entities_union_rels, successes), \
            dtype='float32')
    rhs = sp.coo_matrix((np.ones(successes), (rhs_id, np.arange(successes))), \
            shape = (num_entities_union_rels, successes), \
            dtype='float32')
    rel = sp.coo_matrix((np.ones(successes), (rel_id, np.arange(successes))), \
            shape = (num_entities_union_rels, successes), \
            dtype='float32')
    print 'successes, total: ' + str(successes) + ', ' + str(i) + ', ' + str(successes/i)
    assert successes == len(sentences) == np.shape(lhs)[1] == np.shape(rhs)[1] == np.shape(rel)[1]
    return lhs.tocsr(), rhs.tocsr(), rel.tocsr(), sentences, successes


### main

### open manifest of formatted data
USE_FINAL = True
try:
    m = open(data_path + manifest_file, 'r')
    manifest_list = m.readlines()
    print 'loaded manifest of which data files need to be processed...'
except:
    print 'no manifest file, aborting'
    exit()

### filter Clueweb to include only FB15k-valid triples chunk by chunk
total_successes, total_instances = 0, 0
for chunk in manifest_list:
    chunk = chunk.strip()
    if USE_FINAL: 
        chunk = chunk.replace('processed', 'final') ### use only final files
    try:
        formatted_data_chunk = open(data_path + chunk, 'r')
        print 'processing ' + str(chunk)
    except:
        print 'no \".processed\" file exists, run parse_text_anchors_vandurme.py on a corpus'
        continue
    lhs, rhs, rel, sentences, successes = process(formatted_data_chunk, chunk)
    LHS = sp.hstack((LHS, lhs))
    RHS = sp.hstack((RHS, rhs))
    REL = sp.hstack((REL, rel))
    SENT += sentences 
    total_successes += successes

print 'LHS: ' + str(np.shape(LHS))
print 'RHS: ' + str(np.shape(LHS))
print 'REL: ' + str(np.shape(LHS))

datatyp = 'all'
f = open(data_path + output_prefix + '%s-lhs.pkl' % datatyp, 'w')
g = open(data_path + output_prefix + '%s-rhs.pkl' % datatyp, 'w')
h = open(data_path + output_prefix + '%s-rel.pkl' % datatyp, 'w')
s = open(data_path + output_prefix + '%s-sent.pkl' % datatyp, 'w')
pickle.dump(LHS, f, -1)
pickle.dump(RHS, g, -1)
pickle.dump(REL, h, -1)
pickle.dump(SENT, s, -1)
f.close()
g.close()
h.close()
s.close()
