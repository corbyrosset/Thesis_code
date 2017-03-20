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
    # state.logger.debug('RankLeftFnIdx: rhs=%s, rell=%s, relr=%s, tmp=%s, simi=%s' % (np.shape(rhs), np.shape(rell), np.shape(relr), np.shape(tmp), np.shape(simi)))

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

def TrainFn1Member(margincost, fnsim, embeddings, leftop, rightop, marge=1.0, rel=True, reg=0.001):
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

    # Score of positive triple, the objective could make this lower or higher
    # than the score of a negative triple, depending on which model youre using
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    # Make a negative triple by corrupting left entity
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    # Make another negative triple by corrupting right entity
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    # regularize only relations, since entities are re-normalized to unit L2
    if reg == None:
        cost = costl + costr 
    else:
        cost = costl + costr + (reg/2)*relationl.L2_sqr_norm + (reg/2)*relationr.L2_sqr_norm
    
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
        list_in += [inpon]


    updates = OrderedDict()
    ### update parameters with a particular learning rate
    params = leftop.params + rightop.params ### KG embedding params
    if hasattr(fnsim, 'params'):
        params += KBsim.params
    # updates = sgd(cost, params, learning_rate=lrparams)
    if params:
        updates = sgd(cost, params, learning_rate=lrparams)

    ### update embeddings with a different learning rate
    embedding_params = [embedding.E]
    if type(embeddings) == list:
        embedding_params += [relationl.E] + [relationr.E]

    update_embeddings = sgd(cost,embedding_params, learning_rate=lrembeddings)
    updates.update(update_embeddings)

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
    if rel:
        return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outo), T.mean(outr)],
            updates=updates, on_unused_input='ignore')
    else:
        return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outr)],
            updates=updates, on_unused_input='ignore')
    
def Train1MemberTextAsRel(margincost, KBsim, textsim, KBembeddings, wordembeddings, leftop, rightop, marge=1.0, gamma=0.01, rel=True):
    """
    Here we use the textual embedding *AS* the relation embedding; the
    true relation is never used. 

    score = s(h, r, r_text, t) = ||h + r_text - t|| (e.g. for transE)

    Loss = gamma * [
              max{s(h, r_text, t) - s(h', r_text, t) + marge), 0}
            + max{s(h, r_text, t) - s(h, r_text, t') + marge), 0}
            ]
    if rel == True, add to loss:
        gamma * max{s(h, r_text, t) - s(h, r', t) + marge), 0}

    if rel == True, then we need to rank the positive relation over negative 
    ones. We are still using r_text as the positive relation, but since we
    can't sample "negative texts" easily, we will compare r_text to r'
    where r' is a KB relation embeddings that does NOT appear between 
    entities h and t in the true KB. 

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
    embedding, relationl, relationr = parse_embeddings(KBembeddings)

    # Inputs
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')
    inpr = S.csr_matrix()
    inpl = S.csr_matrix()
    inpo = S.csr_matrix()
    inpln = S.csr_matrix()
    inprn = S.csr_matrix()
    inpon = S.csr_matrix()

    # binary BoW representation of each sentence in the minibatch
    binary_sent = S.csr_matrix() 
    sentinvleng = T.dvector('inverselen') # inverse length of each sentence
    gamma = T.scalar('weightText') 

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T
    # rell = S.dot(relationl.E, inpo).T
    # relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    ### wordembeddings (dim, numWords), binary_sent (batchsize, numWords)
    ### this works, because theano elementwise multiplication is weird :(
    sent_avg = (sentinvleng.T * S.dot(wordembeddings.T, binary_sent.T)).T 
    # sent_avg should now be (minibatchsize, word_dim)
    
    ## similarity of true triple with textual relation instead of KB relation
    simi = KBsim(leftop(lhs, sent_avg), rightop(rhs, sent_avg))
    ## similarity for negative 'left' entity
    similn = KBsim(leftop(lhsn, sent_avg), rightop(rhs, sent_avg))
    ## similirity for negative 'right' entity
    simirn = KBsim(leftop(lhs, sent_avg), rightop(rhsn, sent_avg))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    
    cost = gamma*(costl + costr)

    ##### TODO: perhaps separate outputs into text and KB values as well??

    # List of inputs of the function
    list_in = [lrembeddings, lrparams, inpl, inpr, inpln, inprn, inpon, \
        binary_sent, sentinvleng, gamma]

    if rel: #In addition to ranking text to its relation, rank relation
        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T
        simion = KBsim(leftop(lhs, relln), rightop(rhs, relrn))
        costo, outo = margincost(simi, simion, marge)
        cost += gamma*costo

    updates = OrderedDict()
    ### update parameters with a particular learning rate
    params = leftop.params + rightop.params ### KG embedding params
    if hasattr(KBsim, 'params'):
        params += KBsim.params
    if hasattr(textsim, 'params'):
        params += textsim.params
    # updates = sgd(cost, params, learning_rate=lrparams)
    if params:
        updates = sgd(cost, params, learning_rate=lrparams)

    ### update embeddings of KG (entities + relations) and word embeddings
    embedding_params = [embedding.E, wordembeddings]

    update_embeddings = sgd(cost,embedding_params, learning_rate=lrembeddings)
    updates.update(update_embeddings)

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
    :input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).
    :input binary_sent: variable length vector of integers representing words
                    in the vocabulary (indexes into the wordEmbedding matrix)
    :input sentinvleng: length of binary_sent vector
    :input gamma: weight of cost of textual mention relation

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """

    if rel:
        return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outo), T.mean(outr)],
            updates=updates, on_unused_input='ignore')
    else:
        return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outr)],
            updates=updates, on_unused_input='ignore')

def Train1MemberTextReg(margincost, textsim, KBembeddings, wordembeddings, leftop, rightop, marg_text=2.0, gamma=0.1):
    """
    score(h, r, r_text, t) = ||r - r_text||
    For r' a negative relation, the loss is
    loss = score(h, r', r_text, t) - score(h, r, r_text, t) + marg_text


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
    _, relationl, relationr = parse_embeddings(KBembeddings)

    # Inputs
    lrparams = T.scalar('lrparams')
    lrembeddings = T.scalar('lrembeddings')
    inpo = S.csr_matrix()
    inpon = S.csr_matrix()

    # binary BoW representation of each sentence in the minibatch
    binary_sent = S.csr_matrix() 
    sentinvleng = T.dvector('inverselen') # inverse length of each sentence
    gamma = T.scalar('weightText') 

    # Graph
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    relln = S.dot(relationl.E, inpon).T
    
    sent_avg = (sentinvleng.T * S.dot(wordembeddings.T, binary_sent.T)).T 
    ## sent_avg should now be (minibatchsize, word_dim)
    ## similairty of textual relation embedding and true relation embedding
    textsim_true = textsim(rell, sent_avg)
    ## similarity of textual relation embedding and negative relation embedding
    textsim_neg = textsim(relln, sent_avg)

    costtext, outtext = margincost(textsim_true, textsim_neg, marg_text)
    
    cost = gamma*costtext #MAYBE DONT PUT THIS HERE
    out = outtext

    # List of inputs of the function
    list_in = [lrembeddings, lrparams, inpo, inpon, binary_sent, sentinvleng, gamma]

    updates = OrderedDict()
    ### update parameters with a particular learning rate
    params = leftop.params + rightop.params ### KG embedding params
    if hasattr(textsim, 'params'):
        params += textsim.params
    # updates = sgd(cost, params, learning_rate=lrparams)
    if params:
        updates = sgd(cost, params, learning_rate=lrparams)

    ### update embeddings of KG (entities + relations) and word embeddings
    embedding_params = [wordembeddings]
    if type(KBembeddings) == list:
        embedding_params += [relationl.E] + [relationr.E]

    update_embeddings = sgd(cost,embedding_params, learning_rate=lrembeddings)
    updates.update(update_embeddings)

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
    :input inpon: sparse csr matrix representing the indexes of the
                      negative triplet relation member, shape=(#examples,N
                      [Embeddings]).
    :input binary_sent: variable length vector of integers representing words
                    in the vocabulary (indexes into the wordEmbedding matrix)
    :input sentinvleng: length of binary_sent vector
    :input gamma: weight of cost of textual mention relation

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
            updates=updates, on_unused_input='ignore')

def Train1MemberTextAsRelAndReg():
    raise NotImplementedError

def TrainFnPath(margincost, fnsim, embeddings, leftop, rightop, marge=1.0, reg=0.001):
    """
    Instead of just ranking a positive triple with one relation against a 
    negative triple with one relation, we allow relation paths as long 
    as fnsim is composable, like transE or BilinearDiag. 

    Just like TrainFn1Member, we have only one negative sample per positive
    path. However, relation ranking is not allowed, because it doesn't really
    make sense to rank this "positive path" above a "negative path" - how 
    would you sample the negative path easily?

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    raise NotImplementedError

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
    # regularize only relations, as entities are re-normalized
    ### TODO TODO regularization actually breaks eventually?
    if reg == None:
        cost = costl + costr #+ reg*relationl.L2_sqr_norm 
    else:
        cost = costl + costr + (reg/2)*relationl.L2_sqr_norm + (reg/2)*relationr.L2_sqr_norm
    
    # List of inputs of the function
    list_in = [lrembeddings, lrparams,
            inpl, inpr, inpo, inpln, inprn]

    updates = OrderedDict()
    ### update parameters with a particular learning rate
    params = leftop.params + rightop.params ### KG embedding params
    if hasattr(fnsim, 'params'):
        params += KBsim.params
    # updates = sgd(cost, params, learning_rate=lrparams)
    if params:
        updates = sgd(cost, params, learning_rate=lrparams)

    ### update embeddings with a different learning rate
    embedding_params = [embedding.E]
    if type(embeddings) == list:
        embedding_params += [relationl.E] + [relationr.E]

    update_embeddings = sgd(cost,embedding_params, learning_rate=lrembeddings)
    updates.update(update_embeddings)

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
    return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outr)],
            updates=updates, on_unused_input='ignore')
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

class Baseline1_model():
    '''
        Only consider entity embeddings. The score of a triple (e_h, r, e_t)
        is simply dot(e_h, e_t). This basically just clusters entities that
        co-occur together, which isn't a bad idea. Certainly entities that
        don't co-occur should be ranked below those that do, but that's just
        not enough, hency why this is a baseline that should be beaten. 

        Of course normalize entity embeddings to L2 unit norm
    '''

    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = Unstructured()
            self.rightop = Unstructured()
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                self.embeddings = embeddings
            else:
                try:
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load embeddings')

            ### similarity function of output of left and right ops
            if state.simfn != 'Dot':
                raise ValueError('Baseline1 must use dot similarity')
            self.simfn = Dotsim 
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('Basleline1 must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
            if state.reg != None:
                raise ValueError('no regularization for Baseline1 because there are no relations to regularize')
            self.margincost = eval(state.margincostfunction)
        else:
            try:
                state.logger.info('loading model from file: ' + str(state.loadmodel))
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                raise ValueError('could not load model...')

        # function compilation
        self.trainfunc = TrainFn1Member(self.margincost, self.simfn, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel, reg=state.reg)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            raise ValueError('There are no relations to rank for Baseline1')
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------
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
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load embeddings')
        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 3


            ### similarity function of output of left and right ops
            self.simfn = eval(state.simfn + 'sim') 
            self.margincost = eval(state.margincostfunction)
        else:
            try:
                state.logger.info('loading model from file: ' + str(state.loadmodel))
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                raise ValueError('could not load model...')

        # function compilation
        self.trainfunc = TrainFn1Member(self.margincost, self.simfn, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel, reg=state.reg)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.simfn, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel)
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------
class BilinearDiag_model():
    '''
        For BilinearDiag, the score of a triple is just 
        dot(e_h, r \elemwisemult e_t) because r is a diagonal square matrix, 
        which is really just a vector, so use the same embeddings as TransE
    '''

    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerBilinearDiag()
            self.rightop = Unstructured()
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                relationVec = Embeddings(np.random, state.Nrel, state.ndim, \
                    'relvec')
                ### dummy right relations...
                self.embeddings = [embeddings, relationVec, relationVec]
            else:
                try:
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load embeddings')
        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 3

            ### similarity function of output of left and right ops
            if state.simfn != 'Dot':
                raise ValueError('BilinearDiag must use dotproduct similarity')
            self.simfn = Dotsim 
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('BilinearDiag must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
            self.margincost = eval(state.margincostfunction)
        else:
            try:
                state.logger.info('loading model from file: ' + str(state.loadmodel))
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                raise ValueError('could not load model...')


        # function compilation
        self.trainfunc = TrainFn1Member(self.margincost, self.simfn, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel, reg=state.reg)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.simfn, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel)
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------
class ModelE_model():
    '''
        Model E from "Relation Extraction with Matrix Factorization and 
        Universal Schemas" by Riedel et al ACL 2013

        Each relation is represented by two k-dim vectors, r_h and r_t 

        The score of a triple (e_h, r, e_t) is
        f(e_h, r, e_t) = dot(r_h, e_h) + dot(r_t, e_t)

        We implement this kind of unintuitively as: 
        left = r_h \elemwisemultiply e_h
        right = r_h \elemwisemultiply e_h
        fnsim = T.sum(left, axis = 1) + T.sum(right, axis = 1)

        
        which is maximized when the triple is true. I think there are 
        unit L2 norm constraints on entities, but not sure about 
        relations...

        We cannot rank relations bc we cannot sample negative 
        ones.
    '''

    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerBilinearDiag()
            self.rightop = LayerBilinearDiag()
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                relationVecLeft = Embeddings(np.random, state.Nrel, state.ndim, 'relvecleft')
                relationVecRight = Embeddings(np.random, state.Nrel, state.ndim, 'relvecright')
                self.embeddings = [embeddings, relationVecLeft, relationVecRight]
            else:
                try:
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load model...')

        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 3

            ### similarity function of output of left and right ops
            if state.simfn != 'Sum':
                raise ValueError('Model E must use sum as similarity')
            self.simfn = Sumsim 
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('Model E must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
            self.margincost = eval(state.margincostfunction)
        else:
            try:
                state.logger.info('loading model from file: ' + str(state.loadmodel))
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                raise ValueError('could not load model...')


        # function compilation
        self.trainfunc = TrainFn1Member(self.margincost, self.simfn, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel, reg=state.reg)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.simfn, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel)
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------
class BilinearDiagExtended_model():
    '''
        In the spirit of Model E, make two relation vectors for each KB relation, and then multiply the first one by the first entity elementwise, do the same to the second entity, then take some similarity metric between these two vectors, like dot or L2. 

        That is, the score of a triple is just 
        fnsim((e_h \elemwisemult r_h), (e_t \elemwisemult r_t)) 

        Like Model E, we cannot rank relations bc we cannot sample negative 
        ones.
    '''

    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerBilinearDiag()
            self.rightop = LayerBilinearDiag()
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                relationVecLeft = Embeddings(np.random, state.Nrel, state.ndim, 'relvecleft')
                relationVecRight = Embeddings(np.random, state.Nrel, state.ndim, 'relvecright')
                self.embeddings = [embeddings, relationVecLeft, relationVecRight]
            else:
                try:
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load model...')

        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 3

            ### similarity function of output of left and right ops
            
            self.simfn = eval(state.simfn + 'sim') 
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('BilinearDiagExtended must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
            self.margincost = eval(state.margincostfunction)
        else:
            try:
                state.logger.info('loading model from file: ' + str(state.loadmodel))
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                raise ValueError('could not load model...')


        # function compilation
        self.trainfunc = TrainFn1Member(self.margincost, self.simfn, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel, reg=state.reg)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.simfn, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel)
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------
class TransE_text_model():
    '''
        This uses a word-averaging operator to extract a vector from a textual triple
    '''

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
                ### dummy right relations...
                self.embeddings = [embeddings, relationVec, relationVec]
            else:
                try:
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load model...')

        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 3

            ### Word Embeddings
            self.word_embeddings = WordEmbeddings(state.vocab_size, state.word_dim, wordFile=state.word_file, vocab = state.vocab)

            ### similarity function of output of left and right ops
            self.KBsim = eval(state.simfn + 'sim') 
            self.textsim = eval(state.textsim + 'sim')
            self.margincost = eval(state.margincostfunction)
        else:
            try:
                state.logger.info('loading model from file: ' + str(state.loadmodel))
                f = open(state.loadmodel)
                self.embeddings = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.KBsim = cPickle.load(f)
                self.textsim = cPickle.load(f)
                self.word_embeddings = cPickle.load(f)
                ### TODO: write the word embeddings to file as well
                f.close()
            except:
                raise ValueError('could not load model...')

        self.trainFuncKB = TrainFn1Member(self.margincost, self.KBsim, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel)

        if state.textual_role == 'TextAsRegularizer':
            self.trainFuncText = Train1MemberTextReg(self.margincost, \
                self.textsim, self.embeddings, \
                self.word_embeddings.getEmbeddings(), \
                self.leftop, self.rightop, marg_text=state.marg_text, \
                gamma=state.gamma)
        elif state.textual_role == 'TextAsRelation':
            self.trainFuncText = Train1MemberTextAsRel(self.margincost, \
                self.KBsim, self.textsim, self.embeddings, \
                self.word_embeddings.getEmbeddings(), self.leftop, \
                self.rightop, marge=state.marge, gamma=state.gamma, \
                rel=state.rel)
        elif state.textual_role == 'TextAsRelAndReg':
            self.trainFuncText = Train1MemberTextAsRelAndReg(self.margincost, \
                self.KBsim, self.textsim, self.embeddings, \
                self.word_embeddings.getEmbeddings(), self.leftop, \
                self.rightop, marge=state.marge, gamma=state.gamma, \
                rel=state.rel)
        else:
            raise ValueError("Must supply valid role for a textual instance!")
        
        self.ranklfunc = RankLeftFnIdx(self.KBsim, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        self.rankrfunc = RankRightFnIdx(self.KBsim, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.KBsim, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel)
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------