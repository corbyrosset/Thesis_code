import os
import sys
import time
import copy
import cPickle

import numpy as np
import scipy
import scipy.sparse
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
    randommat = scipy.sparse.lil_matrix((shape[0], shape[1]),
            dtype=theano.config.floatX)
    idx_term = 0
    for idx_ex in range(shape[1]):
        if idx_term == len(listidx):
            idx_term = 0
        randommat[listidx[idx_term], idx_ex] = 1
        idx_term += 1
    return randommat.tocsr()


def load_file(path):
    return scipy.sparse.csr_matrix(cPickle.load(open(path)),
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

    # return trainl, trainr, traino, idxl, idxr, idxo, idxvl, idxvr, idxvo, idxtl, idxtr, idxto, true_triples

    return trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, validridx, validoidx, testlidx, testridx, testoidx, true_triples

def load_clueweb_data(state):
    '''
    From the paper: https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:cr_paper_nips13.pdf


    There are 78,048,372 triplets with textual instances, with the same 
    entities and relations as FB15k, so that's  14,951 entities and 1,345 
    relationships. 

    which were randomly split into  483,142 train, 50,000 valid, and
    59,071 test triples. 

    all of trainx, validx, testx are sparse matrices of 16296 = (14951+1345) by the split size (483,142, etc)
    '''

    print 'not implemented'
    pass