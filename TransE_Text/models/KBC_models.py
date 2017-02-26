import os
import sys
import time
import copy
import cPickle

from Operations import *
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from collections import OrderedDict

# ----------------------------------------------------------------------------

def parse_embeddings(embeddings):
    """
    Utilitary function to parse the embeddings parameter in a normalized way
    for the Structured Embedding [Bordes et al., AAAI 2011] and the Semantic
    Matching Energy [Bordes et al., AISTATS 2012] models.
    """
    if type(embeddings) == list:
        embedding = embeddings[0]
        relationl = embeddings[1]
        relationr = embeddings[2]
    else:
        embedding = embeddings
        relationl = embeddings
        relationr = embeddings
    return embedding, relationl, relationr


# Theano functions creation --------------------------------------------------
def SimFn(fnsim, embeddings, leftop, rightop):
    """
    This function returns a Theano function to measure the similarity score
    for sparse matrices inputs.

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix('inpr')
    inpl = S.csr_matrix('inpl')
    inpo = S.csr_matrix('inpo')
    
    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T ### not used for TransE
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    """
    Theano function inputs.
    :input inpl: sparse csr matrix (representing the indexes of the 'left'
                    entities), shape=(#examples, N [Embeddings]).
    :input inpr: sparse csr matrix (representing the indexes of the 'right'
                    entities), shape=(#examples, N [Embeddings]).
    :input inpo: sparse csr matrix (representing the indexes of the
                    relation member), shape=(#examples, N [Embeddings]).

    Theano function output
    :output simi: matrix of score values.
    """
    return theano.function([inpl, inpr, inpo], [simi],
            on_unused_input='ignore')

def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'right' entities given couples of relation and 'left' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = leftop(lhs, rell)
    simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, relr))
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxo], [simi], on_unused_input='ignore')
    
def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    tmp = rightop(rhs, relr)
    simi = fnsim(leftop(lhs, rell), tmp.reshape((1, tmp.shape[1])))
    """
    Theano function inputs.
    :input idxr: index value of the 'right' member.
    :input idxo: index value of the relation member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxr, idxo], [simi],
            on_unused_input='ignore')

def RankRelFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None):
    """
    This function returns a Theano function to measure the similarity score of
    all relation types given couples of 'left' and 'right' entities (as
    index values).

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxo')
    idxl = T.iscalar('idxl')
    # Graph
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rell = (relationl.E[:, :subtensorspec]).T
        relr = (relationr.E[:, :subtensorspec]).T
    else:
        rell = embedding.E.T
        relr = embedding.E.T
    # hack to prevent a broadcast problem with the Bilinear layer
    if hasattr(leftop, 'forwardrankrel'):
        tmpleft = leftop.forwardrankrel(lhs, rell)
    else:
        tmpleft = leftop(lhs, rell)
    if hasattr(rightop, 'forwardrankrel'):
        tmpright = rightop.forwardrankrel(rhs, relr)
    else:
        tmpright = rightop(lhs, rell)
    simi = fnsim(tmpleft, tmpright)
    """
    Theano function inputs.
    :input idxl: index value of the 'left' member.
    :input idxr: index value of the 'right' member.

    Theano function output.
    :output simi: vector of score values.
    """
    return theano.function([idxl, idxr], [simi],
            on_unused_input='ignore')

def TrainFn1Member(fnsim, embeddings, leftop, rightop, marge=1.0, rel=True):
    """
    This function returns a theano function to perform a training iteration,
    contrasting positive and negative triplets. members are given as sparse
    matrices. For one positive triplet there are two or three (if rel == True)
    negative triplets. To create a negative triplet we replace only one member
    at a time.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    # Negative 'left' member
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    # Negative 'right' member
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    cost = costl + costr
    out = T.concatenate([outl, outr])
    # List of inputs of the function
    list_in = [lrembeddings, lrparams,
            inpl, inpr, inpo, inpln, inprn]
    if rel:
        # If rel is True, we also consider a negative relation member
        inpon = S.csr_matrix()
        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T
        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        cost += costo
        out = T.concatenate([out, outo])
        list_in += [inpon]

    if hasattr(fnsim, 'params'):
        # If the similarity function has some parameters, we update them too.
        
        # updates = sgd(cost, (leftop.params + rightop.params + fnsim.params), learning_rate=lrparams)
        updates = adam(cost, (leftop.params + rightop.params + fnsim.params), learning_rate=lrparams)
    else:
        # updates = sgd(cost, (leftop.params + rightop.params), learning_rate=lrparams)
        updates = adam(cost, (leftop.params + rightop.params), learning_rate=lrparams)

    # gradients_embedding = T.grad(cost, embedding.E)
    # newE = embedding.E - lrembeddings * gradients_embedding
    update_embeddings = sgd(cost, [embedding.E], learning_rate=lrembeddings)
    updates.update(update_embeddings)
    if type(embeddings) == list:
        # If there are different embeddings for the relation member.
        #update_embeddings_l = sgd(cost, [relationl.E], learning_rate=lrparams)
        #updates.update(update_embeddings_l)
        #update_embeddings_r = sgd(cost, [relationr.E], learning_rate=lrparams)
        #updates.update(update_embeddings_r)

        update_embeddings_l = adam(cost, [relationl.E], learning_rate=lrparams)
        updates.update(update_embeddings_l)
        update_embeddings_r = adam(cost, [relationr.E], learning_rate=lrparams)
        updates.update(update_embeddings_r)

    """
    Theano function inputs.
    :input lrembeddings: learning rate for the embeddings.
    :input lrparams: learning rate for the parameters.
    :input inpl: sparse csr matrix representing the indexes of the positive
                 triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive
                 triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive
                 triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative
                  triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative
                  triplet 'right' member, shape=(#examples,N [Embeddings]).
    :opt input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
            updates=updates, on_unused_input='ignore')
    
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------


### TODO:
class TransE_model():

    def __init__(self, state):
        '''define 
            self.leftop, 
            self.rightop, 
            self.embeddings,
            self.relationsVec,
            self.trainfunc,
            self.ranklfunc,
            self.rankrfunc,
        '''
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerTrans()
            self.rightop = Unstructured()
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                relationVec = Embeddings(np.random, state.Nrel, state.ndim, \
                    'relvec')
                self.embeddings = [embeddings, relationVec, relationVec]
            else:
                try:
                    print 'loading embeddings from ' + str(state.loademb)
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    print type(self.embeddings)
                    f.close()
                except:
                    print 'could not load embeddings'
                    exit(1)
        
            assert type(self.embeddings) is list

            ### similarity function of output of left and right ops
            self.simfn = eval(state.simfn + 'sim') 
        else:
            try:
                print 'loading model from file: ' + str(state.loadmodel)
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                print 'could not load model...'
                exit(1)

        # function compilation
        self.trainfunc = TrainFn1Member(self.simfn, self.embeddings, \
            self.leftop, self.rightop, marge=state.marge, rel=state.rel)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.simfn, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel)
        else:
            self.rankrelfunc = None


