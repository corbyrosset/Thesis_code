from __future__ import division
import cPickle as pickle
import numpy as np
import pprint
import ast
import scipy.sparse as sp

'''
    Note for the clueweb dataset defined by the _final files (which were 
    processed to remove punctuation and numbers and such), only 7.937 
    percent of strings are unique! And furthermore, 39.4 percent of those 
    strings were empty after the pre-processing that generated the _final files
'''

data_path = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/'
FB15k_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/'
execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/process_clueweb/'
manifest_file = 'manifest.txt'
output_prefix = 'clueweb_FB15k_'

entity2idx = pickle.load(open(FB15k_path + 'FB15k_entity2idx.pkl', 'r'))
idx2entity = pickle.load(open(FB15k_path + 'FB15k_idx2entity.pkl', 'r'))
num_entities_union_rels = np.max(entity2idx.values()) + 1
USE_FINAL = False #True


### sparse matrices to be output, similar to what FB15_parse.py outputs
LHS = None
RHS = None
REL = None
SENT = []
unique_sent_cntr = 0
unique_sent_map = {}
idx_2_sent_map = {}


def process(formatted_data_chunk, chunk):
    '''take a clueweb data chunk (as file descriptor and title), and output 4 matrices: lhs is a sparse array for left entities, just like it is 
        used for FB15k data. rhs, rel are the same, the only difference is 
        sent, which will be a list of strings corresponding to the textual 
        mentions
    '''
    global unique_sent_cntr
    global unique_sent_map
    global idx_2_sent_map

    successes, failures = 0, 0
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
            failures += 1
            continue ### dictionary error
        if len(text) == 0:
            failures += 1
            continue ### null text error 
        else: # it is valid FB15k triple
            lhs_id.append(entity2idx[leftID])
            rhs_id.append(entity2idx[rightID])
            rel_id.append(entity2idx[relationship])
            ### the number of unique textual instances is not known apriori
            if text not in unique_sent_map:
                unique_sent_map[text] = unique_sent_cntr
                idx_2_sent_map[unique_sent_cntr] = text
                unique_sent_cntr += 1
            sentences.append(unique_sent_map[text])
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
    print '\t(successful conversions, failures, total attempts, percent success) = ' + str(successes) + ', ' + str(failures) + ', ' + str(i) + ', ' + str(successes/i)
    assert successes == len(sentences) == np.shape(lhs)[1] == np.shape(rhs)[1] == np.shape(rel)[1]
    assert len(unique_sent_map.keys()) == (unique_sent_cntr) == len(idx_2_sent_map.keys())
    return lhs.tocsc(), rhs.tocsc(), rel.tocsc(), sentences, successes


###############################################################################
#                     Create/filter the data - only run once!
###############################################################################

### open manifest of formatted data
try:
    m = open(data_path + manifest_file, 'r')
    manifest_list = m.readlines()
    print 'loaded manifest of which data files need to be processed...'
except:
    print 'no manifest file, aborting'
    exit()

## filter Clueweb to include only FB15k-valid triples chunk by chunk
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
    LHS = sp.hstack((LHS, lhs), format='csc')
    RHS = sp.hstack((RHS, rhs), format='csc')
    REL = sp.hstack((REL, rel), format='csc')
    SENT += sentences 
    total_successes += successes

SENT = np.array(SENT)

print '\ndone...'
print 'Number of unique strings: ' + str(unique_sent_cntr)
print 'total successes: ' + str(total_successes)
print 'LHS: ' + str(np.shape(LHS)) + ' ' + str(type(LHS))
print 'RHS: ' + str(np.shape(LHS))
print 'REL: ' + str(np.shape(LHS))
print 'SENT: ' + str(np.shape(SENT))
assert total_successes == len(SENT) == np.shape(LHS)[1] \
    == np.shape(RHS)[1] == np.shape(REL)[1]
assert len(unique_sent_map.keys()) == (unique_sent_cntr) == len(idx_2_sent_map.keys())

datatyp = 'raw'
f = open(data_path + output_prefix + '%s-lhs.pkl' % datatyp, 'w')
g = open(data_path + output_prefix + '%s-rhs.pkl' % datatyp, 'w')
h = open(data_path + output_prefix + '%s-rel.pkl' % datatyp, 'w')
s = open(data_path + output_prefix + '%s-sent.pkl' % datatyp, 'w')
sent2idx = open(data_path + output_prefix + '%s-sent2idx.pkl' % datatyp, 'w')
idx2sent = open(data_path + output_prefix + '%s-idx2sent.pkl' % datatyp, 'w')

pickle.dump(LHS, f, -1)
pickle.dump(RHS, g, -1)
pickle.dump(REL, h, -1)
pickle.dump(SENT, s, -1)
pickle.dump(unique_sent_map, sent2idx, -1)
pickle.dump(idx_2_sent_map, idx2sent, -1)

f.close()
g.close()
h.close()
s.close()
sent2idx.close()
idx2sent.close()
print 'done writing triple components to their respective files'

###############################################################################
#                               Split the data
###############################################################################
### uncomment below to split the data (re -load it)

# train = 28375017
# valid = 37833356

# datatyp = 'raw'
# f = open(data_path + output_prefix + '%s-lhs.pkl' % datatyp, 'r')
# g = open(data_path + output_prefix + '%s-rhs.pkl' % datatyp, 'r')
# h = open(data_path + output_prefix + '%s-rel.pkl' % datatyp, 'r')
# s = open(data_path + output_prefix + '%s-sent.pkl' % datatyp, 'r')
# sent2idx = open(data_path + output_prefix + '%s-sent2idx.pkl' % datatyp, 'r')
# idx2sent = open(data_path + output_prefix + '%s-idx2sent.pkl' % datatyp, 'r')
# LHS = pickle.load(f)
# RHS = pickle.load(g)
# REL = pickle.load(h)
# SENT = pickle.load(s)
# unique_sent_map = pickle.load(sent2idx)
# idx_2_sent_map = pickle.load(idx2sent)
# print 'loaded rhs, lhs, rel, sent, sentence and index maps'

# ### they need to be csc's in order to be fast
# assert isinstance(LHS, sp.csc_matrix)
# assert isinstance(RHS, sp.csc_matrix) 
# assert isinstance(REL, sp.csc_matrix) 
# assert isinstance(SENT, np.ndarray)

# assert 47291696 == len(SENT) == np.shape(LHS)[1] == np.shape(RHS)[1] \
#     == np.shape(REL)[1]
# assert len(unique_sent_map.keys()) == 6194853 == len(idx_2_sent_map.keys())

# order = np.random.permutation(LHS.shape[1])
# LHS = LHS[:, order]
# RHS = RHS[:, order]
# REL = REL[:, order]
# SENT = SENT[order]
# print 'permuted/shuffled data randomly'

# ### save split data
# datatyp = 'raw_train'
# train_lhs = LHS[:, 0:train]
# train_rhs = RHS[:, 0:train]
# train_rel = REL[:, 0:train]
# train_sent = SENT[0:train]
# pickle.dump(train_lhs, open(data_path + output_prefix + '%s-lhs.pkl' % datatyp, 'w'), -1)
# pickle.dump(train_rhs, open(data_path + output_prefix + '%s-rhs.pkl' % datatyp, 'w'), -1)
# pickle.dump(train_rel, open(data_path + output_prefix + '%s-rel.pkl' % datatyp, 'w'), -1)
# pickle.dump(train_sent, open(data_path + output_prefix + '%s-sent.pkl' % datatyp, 'w'), -1)

# datatyp = 'raw_valid'
# valid_lhs = LHS[:, train:valid]
# valid_rhs = RHS[:, train:valid]
# valid_rel = REL[:, train:valid]
# valid_sent = SENT[train:valid]
# pickle.dump(valid_lhs, open(data_path + output_prefix + '%s-lhs.pkl' % datatyp, 'w'), -1)
# pickle.dump(valid_rhs, open(data_path + output_prefix + '%s-rhs.pkl' % datatyp, 'w'), -1)
# pickle.dump(valid_rel, open(data_path + output_prefix + '%s-rel.pkl' % datatyp, 'w'), -1)
# pickle.dump(valid_sent, open(data_path + output_prefix + '%s-sent.pkl' % datatyp, 'w'), -1)

# datatyp = 'raw_test'
# test_lhs = LHS[:, valid:]
# test_rhs = RHS[:, valid:]
# test_rel = REL[:, valid:]
# test_sent = SENT[valid:]
# pickle.dump(test_lhs, open(data_path + output_prefix + '%s-lhs.pkl' % datatyp, 'w'), -1)
# pickle.dump(test_rhs, open(data_path + output_prefix + '%s-rhs.pkl' % datatyp, 'w'), -1)
# pickle.dump(test_rel, open(data_path + output_prefix + '%s-rel.pkl' % datatyp, 'w'), -1)
# pickle.dump(test_sent, open(data_path + output_prefix + '%s-sent.pkl' % datatyp, 'w'), -1)

# f.close()
# g.close()
# h.close()
# s.close()
# sent2idx.close()
# idx2sent.close()
# print 'done splitting data into train, dev, test'
