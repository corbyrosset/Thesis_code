import os
import sys
import time
import copy
import cPickle

import numpy as np
import scipy
import scipy.sparse as sp
import theano

# Utils ----------------------------------------------------------------------
class DD(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z

def negative_samples_filtered(pos_left, pos_rel, pos_right, KB, rels=None):
    """
    This function create a sparse index matrix with the same shape as the 
    training batch representing one negative triple per triple in the 
    train_batch. For a triple in the train_batch, we don't allow its 
    corresponding negative sample to be present in the true_triples. Since we 
    are returning up to sparse binary 3 matrices (one for left entities, right 
    entities, and optionally one for relations), we need to check whether 
    the combination of each negative matrix combined with the other two 
    positive matrices yields any accidental positive examples. 


    :param train_batch: the batch for which we must create a corresponding set of negative samples; their shapes will match. 
    :param true_triples: a set of all true triples, i.e. the knowledge graph
    :param list_idx: list of indices to sample from (default None: it samples from all train_batch.shape[0] indexes, that is, all entities).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """

    num_samples = np.shape(pos_left)[1]
    num_entities = np.shape(pos_left)[0]
    assert np.shape(pos_left)[1] == np.shape(pos_right)[1] 
    assert np.shape(pos_left)[0] == np.shape(pos_right)[0] 
    rows_lhs, cols_lhs, _ = sp.find(pos_left)
    rows_rel, cols_rel, _ = sp.find(pos_rel)
    rows_rhs, cols_rhs, _ = sp.find(pos_right)
    
    ### sample some random indices, will check later
    neg_l_idx = np.random.random_integers(0, high=num_entities-1, \
        size=(num_samples))
    neg_r_idx = np.random.random_integers(0, high=num_entities-1, \
        size=(num_samples))
    if rels:
        neg_rel_idx = np.random.random_integers(0, high=rels-1, \
            size=(num_samples))
    
    ### check if (rows_lhs, rows_rel, neg_r_idx) leads to conflicts
    check_right = np.concatenate([rows_lhs, rows_rel, neg_r_idx]).reshape(3,num_samples).T
    resample_cntr = 0
    for i, row in enumerate(check_right):
        row = tuple(row)
        while True:
            if row not in KB:
                break
            neg_r_idx[i] = np.random.random_integers(0, high=num_entities-1, \
                size=1)
            row = (row[0], row[1], neg_r_idx[i])
            resample_cntr += 1
        assert (row[0], row[1], neg_r_idx[i]) not in KB
    # print 'resampled right %s times' % resample_cntr

    ### check if (neg_l_idx, rows_rel, rows_rhs) leads to conflicts
    check_left = np.concatenate([neg_l_idx, rows_rel, rows_rhs]).reshape(3,num_samples).T
    resample_cntr = 0
    for i, row in enumerate(check_left):
        row = tuple(row)
        while True:
            if row not in KB:
                break
            neg_l_idx[i] = np.random.random_integers(0, high=num_entities-1, \
                size=1)
            row = (neg_l_idx[i], row[1], row[2])
            resample_cntr += 1
        assert (neg_l_idx[i], row[1], row[2]) not in KB
    # print 'resampled left %s times' % resample_cntr

    ### check if (rows_lhs, neg_rel_idx, rows_rhs) leads to conflicts
    if rels:
        check_rels = np.concatenate([rows_lhs, neg_rel_idx, rows_rhs]).reshape(3, num_samples).T
        resample_cntr = 0
        for i, row in enumerate(check_rels):
            row = tuple(row)
            while True:
                if row not in KB:
                    break
                neg_rel_idx[i] = np.random.random_integers(0, high=rels-1, \
                    size=1)
                row = (row[0], neg_rel_idx[i], row[2])
                resample_cntr += 1
            assert (row[0], neg_rel_idx[i], row[2]) not in KB
        # print 'resampled relations %s times' % resample_cntr

    if rels:
        return expand_to_mat(neg_l_idx.tolist(), num_entities), expand_to_mat(neg_rel_idx.tolist(), rels), expand_to_mat(neg_r_idx.tolist(), num_entities)
    else:
        return expand_to_mat(neg_l_idx.tolist(), num_entities), expand_to_mat(neg_r_idx.tolist(), num_entities)

def create_random_mat(shape, listidx=None):
    """
    This function create a random sparse index matrix with a given shape. It
    is useful to create negative triplets.

    :param shape: shape of the desired sparse matrix.
    :param listidx: list of index to sample from (default None: it samples from
                    all shape[0] indexes).

    :note: if shape[1] > shape[0], it loops over the shape[0] indexes.
    """
    if listidx is None:
        listidx = np.arange(shape[0])
    listidx = listidx[np.random.permutation(len(listidx))]
    randommat = sp.lil_matrix((shape[0], shape[1]),
            dtype=theano.config.floatX)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()

def binarize_sentences(sents, vocabSize):
    '''
    take as input a list of N sentences, each represented as a list of 
    word_indexes (where word_index ranges from 0 to the vocab size, V). 
    Convert this to a row-sparse scipy array of size N by V, which contains a 
    1 in position (i, j) iff sentence i contained word j. Also return the 
    inverse lengths of each sentence. 
    '''
    mat = sp.lil_matrix((len(sents), vocabSize),
            dtype=theano.config.floatX)
    lens = []
    for i, sent in enumerate(sents):
        for j in sent:
            mat[i, j] = 1
        lens.append(1/float(len(sent))) # inverse of sentence length
    return mat.tocsr(), lens

def expand_to_mat(lst, sz):
    '''
    take as input a list of N values, each values on the range [0, sz)
    and construct a sz by N sparse binary matrix where mat[i, j] = 1
    iff the j'th element of the list is i. 
    '''
    V = np.ones((len(lst)))
    J = np.arange(0, len(lst))
    I = np.array(lst)
    return sp.coo_matrix((V,(I,J)),shape=(sz,len(lst))).tocsc()

def load_file(path):
    return sp.csc_matrix(cPickle.load(open(path)),
            dtype=theano.config.floatX)


def convert2idx(spmat):
    rows, cols = spmat.nonzero()
    return rows[np.argsort(cols)]


def load_FB15k_data(state):
    '''
    From the paper: https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:cr_paper_nips13.pdf


    There are 592,213 triplets with 14,951 entities and 1,345 relationships 
    which were randomly split into  483,142 train, 50,000 valid, and
    59,071 test triples. The relationships occur at the bottom of every
    column vector, hence why "traino[-state.Nrel:, :]"

    all of trainx, validx, testx are sparse matrices of 16296 = (14951+1345) by the split size (483,142, etc)
    '''
    start = time.clock()
    # Positives
    trainl = load_file(state.datapath + state.dataset + '-train-lhs.pkl')
    trainr = load_file(state.datapath + state.dataset + '-train-rhs.pkl')
    traino = load_file(state.datapath + state.dataset + '-train-rel.pkl')
    # if state.op == 'SE' or state.op == 'TransE':
    traino = traino[-state.Nrel:, :] ### take bottom Nrel relations?
    # elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
    #     trainl = trainl[:state.Nsyn, :]
    #     trainr = trainr[:state.Nsyn, :]
    #     traino = traino[-state.Nrel:, :]

    # Valid set
    validl = load_file(state.datapath + state.dataset + '-valid-lhs.pkl')
    validr = load_file(state.datapath + state.dataset + '-valid-rhs.pkl')
    valido = load_file(state.datapath + state.dataset + '-valid-rel.pkl')
    # if state.op == 'SE' or state.op == 'TransE':
    valido = valido[-state.Nrel:, :]
    # elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
    #     validl = validl[:state.Nsyn, :]
    #     validr = validr[:state.Nsyn, :]
    #     valido = valido[-state.Nrel:, :]


    # Test set
    testl = load_file(state.datapath + state.dataset + '-test-lhs.pkl')
    testr = load_file(state.datapath + state.dataset + '-test-rhs.pkl')
    testo = load_file(state.datapath + state.dataset + '-test-rel.pkl')
    # if state.op == 'SE' or state.op == 'TransE':
    testo = testo[-state.Nrel:, :]
    # elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
    #     testl = testl[:state.Nsyn, :]
    #     testr = testr[:state.Nsyn, :]
    #     testo = testo[-state.Nrel:, :]

    # Index conversion 
    ### train indices
    idxl = convert2idx(trainl)
    idxr = convert2idx(trainr)
    idxo = convert2idx(traino)
    if state.ntrain == 'all':
        trainlidx = idxl
        trainridx = idxr
        trainoidx = idxo
    else:
        trainlidx = idxl[:state.ntrain]
        trainridx = idxr[:state.ntrain]
        trainoidx = idxo[:state.ntrain]

    ### valid indices
    idxvl = convert2idx(validl)
    idxvr = convert2idx(validr)
    idxvo = convert2idx(valido)
    if state.nvalid == 'all':
        validlidx = idxvl
        validridx = idxvr
        validoidx = idxvo
    else:
        validlidx = idxvl[:state.nvalid]
        validridx = idxvr[:state.nvalid]
        validoidx = idxvo[:state.nvalid]

    ### test indices
    idxtl = convert2idx(testl)
    idxtr = convert2idx(testr)
    idxto = convert2idx(testo)
    if state.ntest == 'all':
        testlidx = idxtl
        testridx = idxtr
        testoidx = idxto
    else:
        testlidx = idxtl[:state.ntest]
        testridx = idxtr[:state.ntest]
        testoidx = idxto[:state.ntest]

    true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T
    KB = set([tuple(i) for i in true_triples]) ### {(head, rel, tail)}
    assert len(KB) == np.shape(true_triples)[0]

    elapsed = time.clock()
    elapsed = elapsed - start
    print "loaded FB15k data in: %s seconds" % elapsed
    
    return trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, validridx, validoidx, testlidx, testridx, testoidx, true_triples, KB

def load_FB15k_Clueweb_data(state):
    '''
    From the paper: https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:cr_paper_nips13.pdf


    There are 78,048,372 triplets with textual instances, with the same 
    entities and relations as FB15k, so that's  14,951 entities and 1,345 
    relationships. 

    which were randomly split into  483,142 train, 50,000 valid, and
    59,071 test triples. 

    all of trainx, validx, testx are sparse matrices of 16296 = (14951+1345) by the split size (483,142, etc)

    - text_per_triple_cntr: The count total number of textual instances for each triple
    - unique_text_per_triple: The set of unique text mentions for each triple
    - triple_per_text: The converse of unique_text_per_triple, i.e. the set
         of triples labeled on each unique textual instance. This is scary

    Note, textual triples will not be evaluated on, not during training nor testing. 
    '''

    data_path = state.datapath
    start = time.clock()
    idxl = None
    idxr = None
    idxo = None
    idxvl = None
    idxvr = None
    idxvo = None
    idxtl = None
    idxtr = None
    idxto = None
    text_train = None
    text_valid = None
    text_test = None
    sent2idx = None
    idx2sent = None
    text_per_triple_cntr = None
    unique_text_per_triple = None
    triple_per_text = None

    datatyp = 'train'
    lhs_train = load_file(data_path + 'clueweb_FB15k_filtered_%s-lhs.pkl' % datatyp)
    rhs_train = load_file(data_path + 'clueweb_FB15k_filtered_%s-rhs.pkl' % datatyp)
    rel_train = load_file(data_path + 'clueweb_FB15k_filtered_%s-rel.pkl' % datatyp)
    text_train = np.array(cPickle.load(open(data_path + 'clueweb_FB15k_filtered_%s-sent.pkl' % datatyp)))[:state.numTextTrain]
    rel_train = rel_train[-state.Nrel:, :]
    ### TODO: reformat input data so you don't need to do '[0, :].tolist()'
    idxl = expand_to_mat(convert2idx(lhs_train)[:state.numTextTrain], \
            state.Nent)
    idxr = expand_to_mat(convert2idx(rhs_train)[:state.numTextTrain], \
            state.Nent)
    idxo = expand_to_mat(convert2idx(rel_train)[:state.numTextTrain], \
            state.Nrel)

    # datatyp = 'valid'
    # lhs_valid = load_file(data_path + 'clueweb_FB15k_%s-lhs.pkl' % datatyp)
    # rhs_valid = load_file(data_path + 'clueweb_FB15k_%s-rhs.pkl' % datatyp)
    # rel_valid = load_file(data_path + 'clueweb_FB15k_%s-rel.pkl' % datatyp)
    # text_valid = load_file(data_path + 'clueweb_FB15k_%s-sent.pkl' % datatyp)
    # rel_valid = rel_valid[-state.Nrel:, :]
    # idxvl = convert2idx(lhs_valid)
    # idxvr = convert2idx(rhs_valid)
    # idxvo = convert2idx(rel_valid)

    # datatyp = 'test'
    # lhs_test = load_file(data_path + 'clueweb_FB15k_%s-lhs.pkl' % datatyp)
    # rhs_test = load_file(data_path + 'clueweb_FB15k_%s-rhs.pkl' % datatyp)
    # rel_test = load_file(data_path + 'clueweb_FB15k_%s-rel.pkl' % datatyp)
    # text_test = load_file(data_path + 'clueweb_FB15k_%s-sent.pkl' % datatyp)
    # rel_test = rel_test[-state.Nrel:, :]
    # idxtl = convert2idx(lhs_test)
    # idxtr = convert2idx(rhs_test)
    # idxto = convert2idx(rel_test)
    
    ### index over unique sentences
    datatyp = 'all'
    # sent2idx = cPickle.load(open(data_path + 'clueweb_FB15k_%s-sent2idx.pkl' % datatyp, 'r'))
    idx2sent = cPickle.load(open(data_path + 'clueweb_FB15k_%s-idx2sent.pkl' % datatyp, 'r'))

    ### counters of associations between text and triples
    # text_per_triple_cntr = cPickle.load(open(data_path + \
    #    'grouped_FB15k_clueweb_triples-counts.pkl', 'r'))
    # unique_text_per_triple = cPickle.load(open(data_path + \
    #    'grouped_FB15k_clueweb_triples-text_sets.pkl', 'r'))
    triple_per_text = cPickle.load(open(data_path + \
        'grouped_FB15k_clueweb_triples-triple_per_text_count.pkl', 'r'))


    elapsed = time.clock()
    elapsed = elapsed - start

    print "loaded clueweb data in: %s seconds" % elapsed
    print "size of clueweb data: %s bytes" % (sys.getsizeof(idxl) \
    + sys.getsizeof(idxr) + sys.getsizeof(idxo) + sys.getsizeof(idxvl) \
    + sys.getsizeof(idxvr) + sys.getsizeof(idxvo) + sys.getsizeof(idxtl) \
    + sys.getsizeof(idxtr) + sys.getsizeof(idxto) + sys.getsizeof(sent2idx) \
    + sys.getsizeof(idx2sent) + sys.getsizeof(text_per_triple_cntr) \
    + sys.getsizeof(unique_text_per_triple) + sys.getsizeof(triple_per_text))
    print '\tidxl: ' + str(sys.getsizeof(idxl)) + ' bytes'
    print '\tidxr: ' + str(sys.getsizeof(idxr)) + ' bytes'
    print '\tidxo: ' + str(sys.getsizeof(idxo)) + ' bytes'
    print '\tidxvl: ' + str(sys.getsizeof(idxvl)) + ' bytes'
    print '\tidxvr: ' + str(sys.getsizeof(idxvr)) + ' bytes'
    print '\tidxvo: ' + str(sys.getsizeof(idxvo)) + ' bytes'
    print '\tidxtl: ' + str(sys.getsizeof(idxtl)) + ' bytes'
    print '\tidxtr: ' + str(sys.getsizeof(idxtr)) + ' bytes'
    print '\tidxto: ' + str(sys.getsizeof(idxto)) + ' bytes'
    print '\tsent2idx: ' + str(sys.getsizeof(sent2idx)) + ' bytes'
    print '\tidx2sent: ' + str(sys.getsizeof(idx2sent)) + ' bytes'
    print '\ttext_per_triple_cntr: ' + \
        str(sys.getsizeof(text_per_triple_cntr)) + ' bytes'
    print '\tunique_text_per_triple: ' + \
        str(sys.getsizeof(unique_text_per_triple)) + ' bytes'
    print '\ttriple_per_text: ' + str(sys.getsizeof(triple_per_text)) + \
        ' bytes'

    return idxl, idxr, idxo, text_train, idxvl, idxvr, idxvo, text_valid, \
        idxtl, idxtr, idxto, text_test, text_per_triple_cntr, \
        unique_text_per_triple, triple_per_text, sent2idx, idx2sent

