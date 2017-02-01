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
    # Positives
    trainl = load_file(state.datapath + state.dataset + '-train-lhs.pkl')
    trainr = load_file(state.datapath + state.dataset + '-train-rhs.pkl')
    traino = load_file(state.datapath + state.dataset + '-train-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        traino = traino[-state.Nrel:, :] ### take bottom Nrel relations?
    # elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
    #     trainl = trainl[:state.Nsyn, :]
    #     trainr = trainr[:state.Nsyn, :]
    #     traino = traino[-state.Nrel:, :]

    # Valid set
    validl = load_file(state.datapath + state.dataset + '-valid-lhs.pkl')
    validr = load_file(state.datapath + state.dataset + '-valid-rhs.pkl')
    valido = load_file(state.datapath + state.dataset + '-valid-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        valido = valido[-state.Nrel:, :]
    # elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
    #     validl = validl[:state.Nsyn, :]
    #     validr = validr[:state.Nsyn, :]
    #     valido = valido[-state.Nrel:, :]


    # Test set
    testl = load_file(state.datapath + state.dataset + '-test-lhs.pkl')
    testr = load_file(state.datapath + state.dataset + '-test-rhs.pkl')
    testo = load_file(state.datapath + state.dataset + '-test-rel.pkl')
    if state.op == 'SE' or state.op == 'TransE':
        testo = testo[-state.Nrel:, :]
    # elif state.op =='Bi' or state.op == 'Tri'or state.op == 'TATEC':
    #     testl = testl[:state.Nsyn, :]
    #     testr = testr[:state.Nsyn, :]
    #     testo = testo[-state.Nrel:, :]

    # Index conversion
    trainlidx = convert2idx(trainl)[:state.neval]
    trainridx = convert2idx(trainr)[:state.neval]
    trainoidx = convert2idx(traino)[:state.neval]
    validlidx = convert2idx(validl)[:state.neval]
    validridx = convert2idx(validr)[:state.neval]
    validoidx = convert2idx(valido)[:state.neval]
    testlidx = convert2idx(testl)[:state.neval]
    testridx = convert2idx(testr)[:state.neval]
    testoidx = convert2idx(testo)[:state.neval]
    
    idxl = convert2idx(trainl)
    idxr = convert2idx(trainr)
    idxo = convert2idx(traino)
    idxtl = convert2idx(testl)
    idxtr = convert2idx(testr)
    idxto = convert2idx(testo)
    idxvl = convert2idx(validl)
    idxvr = convert2idx(validr)
    idxvo = convert2idx(valido)
    true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T

    return trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, validridx, validoidx, testlidx, testridx, testoidx, true_triples