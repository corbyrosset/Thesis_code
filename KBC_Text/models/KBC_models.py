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

    In my opinion, this is very stupid

    """
    if type(embeddings) == list and len(embeddings) == 3:
        return embeddings[0], embeddings[1], embeddings[2]
    elif type(embeddings) == list and len(embeddings) == 4:
        return embeddings[0], embeddings[1], embeddings[2], embeddings[3]
    else:
        embedding = embeddings
        relationl = embeddings
        relationr = embeddings
    return embedding, relationl, relationr


# Theano functions creation --------------------------------------------------
# def SimFn(fnsim, embeddings, leftop, rightop):
#     """
#     This function returns a Theano function to measure the similarity score
#     for sparse matrices inputs.

#     :param fnsim: similarity function (on Theano variables).
#     :param embeddings: an Embeddings instance.
#     :param leftop: class for the 'left' operator.
#     :param rightop: class for the 'right' operator.
#     """
#     embedding, relationl, relationr = parse_embeddings(embeddings)

#     # Inputs
#     inpr = S.csr_matrix('inpr')
#     inpl = S.csr_matrix('inpl')
#     inpo = S.csr_matrix('inpo')
    
#     # Graph
#     lhs = S.dot(embedding.E, inpl).T
#     rhs = S.dot(embedding.E, inpr).T
#     rell = S.dot(relationl.E, inpo).T
#     relr = S.dot(relationr.E, inpo).T ### not used for TransE
#     simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
#     """
#     Theano function inputs.
#     :input inpl: sparse csr matrix (representing the indexes of the 'left'
#                     entities), shape=(#examples, N [Embeddings]).
#     :input inpr: sparse csr matrix (representing the indexes of the 'right'
#                     entities), shape=(#examples, N [Embeddings]).
#     :input inpo: sparse csr matrix (representing the indexes of the
#                     relation member), shape=(#examples, N [Embeddings]).

#     Theano function output
#     :output simi: matrix of score values.
#     """
#     return theano.function([inpl, inpr, inpo], [simi],
#             on_unused_input='ignore')

def RankRightFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None, modelType=None):
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
    if modelType == 'STransE':
        embedding, embeddingsModifiersLeft, embeddingsModifiersRight, relations = parse_embeddings(embeddings)
        relationl = relations
        relationr = relations
    else:
        embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxl = T.iscalar('idxl')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        rhs = (embedding.E[:, :subtensorspec]).T
        if modelType == 'STransE':
            rmod = (embeddingsModifiersRight.E[:, :subtensorspec]).T
    else:
        rhs = embedding.E.T
        if modelType == 'STransE':
            rmod = embeddingsModifiersRight.E.T

    if modelType == 'STransE':
        lmod = (embeddingsModifiersLeft.E[:, idxl]).reshape((1, embeddingsModifiersLeft.D))
    
    lhs = (embedding.E[:, idxl]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))

    if modelType == 'STransE':
        tmp = leftop(lhs, lmod, rell)
        simi = fnsim(tmp.reshape((1, tmp.shape[1])), rightop(rhs, rmod))
    else:
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

def RankLeftFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None, modelType=None):
    """
    This function returns a Theano function to measure the similarity score of
    all 'left' entities given couples of relation and 'right' entities (as
    index values).

    TODO: we are handling STransE as a special case (bc it has 4 embeddings 
    parameters instead of 3), which is annoying. But there is something 
    even worse, which is that for every setting of idxo and idxr, we have
    to re-compute all the lhs and rhs operations for STransE, which involve
    ~cubic time matrix operations. This is expensive, and it applies to 
    Bilinear as well: that's why it takes 12+ hours to evaluate it on test data
    instead of 30 mins. Solution: find a way to pre-compute, store, and lookup
    the lhs, rhs, and rel operations in each of RankLeftFnIdx, RankRightFnIdx, 
    RankRelFnIdx respectively. This will have to handled on a per-model basis,
    which won't be elegant. For now, we just have it working the naive way. 

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    :param subtensorspec: only measure the similarity score for the entities
                          corresponding to the first subtensorspec (int)
                          entities of the embedding matrix (default None: all
                          entities).
    """
    if modelType == 'STransE':
        embedding, embeddingsModifiersLeft, embeddingsModifiersRight, relations = parse_embeddings(embeddings)
        relationl = relations
        relationr = relations
    else:
        embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    idxr = T.iscalar('idxr')
    idxo = T.iscalar('idxo')
    # Graph
    if subtensorspec is not None:
        # We compute the score only for a subset of entities
        lhs = (embedding.E[:, :subtensorspec]).T
        if modelType == 'STransE':
            lmod = (embeddingsModifiersLeft.E[:, :subtensorspec]).T
    else:
        lhs = embedding.E.T
        if modelType == 'STransE':
            lmod = embeddingsModifiersLeft.E.T

    if modelType == 'STransE':
        rmod = (embeddingsModifiersRight.E[:, idxr]).reshape((1, embeddingsModifiersRight.D))

    rhs = (embedding.E[:, idxr]).reshape((1, embedding.D))
    rell = (relationl.E[:, idxo]).reshape((1, relationl.D))
    relr = (relationr.E[:, idxo]).reshape((1, relationr.D))
    
    if modelType == 'STransE':
        tmp = rightop(rhs, rmod)
        simi = fnsim(leftop(lhs, lmod, rell), tmp.reshape((1, tmp.shape[1])))
    else:
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

def RankRelFnIdx(fnsim, embeddings, leftop, rightop, subtensorspec=None, modelType=None):
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
    if modelType == 'STransE':
        embedding, embeddingsModifiersLeft, embeddingsModifiersRight, relations = parse_embeddings(embeddings)
        relationl = relations
        relationr = relations
    else:
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

    if modelType == 'STransE':
        rmod = (embeddingsModifiersRight.E[:, idxr]).reshape((1, embeddingsModifiersRight.D))
        lmod = (embeddingsModifiersLeft.E[:, idxl]).reshape((1, embeddingsModifiersLeft.D))

    # hack to prevent a broadcast problem with the Bilinear layer
    # if hasattr(leftop, 'forwardrankrel'):
    #     tmpleft = leftop.forwardrankrel(lhs, rell)
    # else:
    if modelType == 'STransE':
        tmpleft = leftop(lhs, lmod, rell)
    else:
        tmpleft = leftop(lhs, rell)


    # if hasattr(rightop, 'forwardrankrel'):
    #     tmpright = rightop.forwardrankrel(rhs, relr)
    # else:
    if modelType == 'STransE':
        tmpright = rightop(rhs, rmod)
    else:
        tmpright = rightop(rhs, relr)


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
        params += fnsim.params
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

def TrainFn1MemberSTransE(margincost, fnsim, embeddings, leftop, rightop, \
    marge=1.0, rel=True, reg=0.001):
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
    :param modifierop: an operator on the entity embeddings, see STransE paper.
    :param marge: marge for the cost function.
    :param rel: boolean, if true we also contrast w.r.t. a negative relation
                member.
    """
    ### entity embedding, matrix for left entity, matrix for right entity, 
    ### relation embeddings
    embedding, embeddingsModifiersLeft, embeddingsModifiersRight, relations = parse_embeddings(embeddings)

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
    rell = S.dot(relations.E, inpo).T
    relr = S.dot(relations.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T

    # Embedding Modifiers
    lmod = S.dot(embeddingsModifiersLeft.E, inpl).T
    rmod = S.dot(embeddingsModifiersRight.E, inpr).T

    # Score of positive triple, the objective could make this lower or higher
    # than the score of a negative triple, depending on which model youre using
    simi = fnsim(leftop(lhs, lmod, rell), rightop(rhs, rmod))
    # Make a negative triple by corrupting left entity
    similn = fnsim(leftop(lhsn, lmod, rell), rightop(rhs, rmod))
    # Make another negative triple by corrupting right entity
    simirn = fnsim(leftop(lhs, lmod, rell), rightop(rhsn, rmod))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)
    # regularize only relations, since entities are re-normalized to unit L2
    if reg == None:
        cost = costl + costr 
    else:
        cost = costl + costr + (reg*relations.L2_sqr_norm)
    
    # List of inputs of the function
    list_in = [lrembeddings, lrparams,
            inpl, inpr, inpo, inpln, inprn]
    if rel:
        # If rel is True, we also consider a negative relation member
        inpon = S.csr_matrix()
        reln = S.dot(relations.E, inpon).T
        simion = fnsim(leftop(lhs, lmod, reln), rightop(rhs, rmod))
        costo, outo = margincost(simi, simion, marge)
        cost += costo
        list_in += [inpon]


    updates = OrderedDict()
    ### update parameters with a particular learning rate
    params = leftop.params + rightop.params ### KG embedding params
    if hasattr(fnsim, 'params'):
        params += fnsim.params
    # updates = sgd(cost, params, learning_rate=lrparams)
    if params:
        updates = sgd(cost, params, learning_rate=lrparams)

    ### update embeddings with a different learning rate
    embedding_params = [embedding.E, embeddingsModifiersLeft.E, embeddingsModifiersRight.E, relations.E]
    update_embeddings = sgd(cost, embedding_params, learning_rate=lrembeddings)
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

    Also, don't regularize the relation emb in this function, since they will
    be regularized in the KB training functions

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

def Train1MemberTextAsRelAndReg(margincost, KBsim, textsim, KBembeddings, wordembeddings, leftop, rightop, marge=1.0, marg_text=1.0, gamma=0.01, rel=False):
    """
    Here we combine the TextAsRegularizer and TextAsRelation paradigms above 
    into one cost function. We use the textual embedding *AS* the relation 
    embedding; and we also use the relation embeddings as something that the 
    textual embedding should be close to

    score1(h, r, r_text, t) = ||h + r_text - t|| (e.g. for transE)
    score2(h, r, r_text, t) = ||r - r_text||

    Loss1 = gamma * [
              max{s(h, r_text, t) - s(h', r_text, t) + marge), 0}
            + max{s(h, r_text, t) - s(h, r_text, t') + marge), 0}
            ]
    For r' a negative relation, the loss is
    Loss2 = score(h, r', r_text, t) - score(h, r, r_text, t) + marg_text

    if rel == True, add to loss:
        gamma * max{s(h, r_text, t) - s(h, r', t) + marge), 0}

    if rel == True, then we need to rank the positive relation over negative 
    ones. We are still using r_text as the positive relation, but since we
    can't sample "negative texts" easily, we will compare r_text to r'
    where r' is a KB relation embeddings that does NOT appear between 
    entities h and t in the true KB. 

    Also, don't regularize the relation emb in this function, since they will
    be regularized in the KB training functions

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
    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T
    relln = S.dot(relationl.E, inpon).T

    ### wordembeddings (dim, numWords), binary_sent (batchsize, numWords)
    ### this works, because theano elementwise multiplication is weird :(
    sent_avg = (sentinvleng.T * S.dot(wordembeddings.T, binary_sent.T)).T 
    # sent_avg should now be (minibatchsize, word_dim)
    
    ### Text as Relation
    # similarity of true triple with textual relation instead of KB relation
    simi = KBsim(leftop(lhs, sent_avg), rightop(rhs, sent_avg))
    ## similarity for negative 'left' entity
    similn = KBsim(leftop(lhsn, sent_avg), rightop(rhs, sent_avg))
    # similirity for negative 'right' entity
    simirn = KBsim(leftop(lhs, sent_avg), rightop(rhsn, sent_avg))
    costl, outl = margincost(simi, similn, marge)
    costr, outr = margincost(simi, simirn, marge)

    ### Text as Regularizer
    textsim_true = textsim(rell, sent_avg)
    # similarity of textual relation embedding and negative relation embedding
    textsim_neg = textsim(relln, sent_avg)
    costtext, outtext = margincost(textsim_true, textsim_neg, marg_text)
    
    ### Combine Costs
    cost = gamma*(costl + costr + costtext)

    ##### TODO: perhaps separate outputs into text and KB values as well??

    # List of inputs of the function
    list_in = [lrembeddings, lrparams, inpl, inpr, inpo, inpln, inprn, inpon, \
        binary_sent, sentinvleng, gamma]

    if rel: #In addition to ranking text to its relation, rank relation
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
        return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outo), T.mean(outr), T.mean(outtext)],
            updates=updates, on_unused_input='ignore')
    else:
        return theano.function(list_in, [T.mean(cost), T.mean(outl), T.mean(outr), T.mean(outtext)],
            updates=updates, on_unused_input='ignore')

def TrainFnPath(compop, use_horn_path, initial_vecs, margincost, fnsim, embeddings, marge=1.0, reg=0.001):
    """
    Instead of just ranking a positive triple with one relation against a 
    negative triple with one relation, we allow relation paths as long 
    as fnsim is composable, like transE or BilinearDiag. 

    Just like TrainFn1Member, we have only one negative sample per positive
    path. However, relation ranking is not allowed, because it doesn't really
    make sense to rank this "positive path" above a "negative path" - how 
    would you sample the negative path easily?

    :param compop: path composition operator
    :param use_horn_path: boolean whether we use horn (cycle-constrained) paths or not 
    :param initial_vecs: if horn paths, what are the initial vectors - 0s or 1s
    :param margincost: how to compute margin cost
    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param marge: marge for the cost function.
    :param reg: whether to regularize and with what weight
    """

    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    lrembeddings = T.scalar('lrembeddings')
    lrparams = T.scalar('lrparams')
    inpo = S.csr_matrix()
    idxs_per_path = T.imatrix("indexes") # N paths (rows), each length L
    # List of inputs of the function
    list_in = [lrembeddings, lrparams, inpo, idxs_per_path]

    # Graph
    rel = S.dot(relationl.E, inpo).T
    
    # Symbolic description of the result
    if not use_horn_path:
        assert initial_vecs == None
        inpl = S.csr_matrix()
        inpr = S.csr_matrix()
        inprn = S.csr_matrix()
        list_in += [inpl, inpr, inprn]

        lhs = S.dot(embedding.E, inpl).T # initial vecs!
        rhs = S.dot(embedding.E, inpr).T
        rhsn = S.dot(embedding.E, inprn).T

        ### we want to ignore previous steps of the outputs.
        ### idxs_per_path is a matrix of N paths (rows), each length L
        ### we are mapping each row of idxs_per_path to a D-dim vector 
        ### embedding of the path. Initial_vecs is N by D, the initial 
        ### embedding of the path (the start entity vector in non-horn paths). 
        ### rels is the D by numRels embedding matrix of relationships, fixed 

        ### because these are not horn paths, initial vectors must be supplied 
        ### by user during training
        pathvec, _ = theano.map(
                        fn= compop, #compose_TransE,
                        sequences=[idxs_per_path, lhs],
                        non_sequences=[rel])
        ### for non-horn paths, we care about bookended entities, so it's 
        ### the same as before:
        simi = fnsim(pathvec, rhs) ### ONLY THIS MAKES SENSE TO DO
        # MAKES NOT SENSE TO USE A Negative 'left' member
        # similn = fnsim(leftop(lhsn, pathvec), rightop(rhs, pathvec))
        # Negative 'right' member
        simirn = fnsim(pathvec, rhsn)
        # costl, outl = margincost(simi, similn, marge)
        cost, out = margincost(simi, simirn, marge)
        # regularize only relations, as entities are re-normalized
        ### TODO TODO regularization actually breaks eventually?

    else:
        assert initial_vecs is not None
        # paths start at first relation, compose relations until finished.
        # final result should be close to the intended cycle-completing rel

        # consider a negative relation member
        inpon = S.csr_matrix()
        listin += [inpon]

        reln = S.dot(relationl.E, inpon).T

        ### be mindful what to set initial vectors to - 0s for transE, 
        ### 1s for bilinear
        if initial_vecs == 0:
            initial_vecs = np.zeros((idxs_per_path.shape[0], relationl.D)
            , dtype=theano.config.floatX) # num paths by dimension
        elif initial_vecs == 1:
            initial_vecs = np.ones((idxs_per_path.shape[0], relationl.D)
            , dtype=theano.config.floatX)
        else: 
            raise ValueError("invalid value for initial vectors, must be identity element in the ring")
        initial_vecs_fixed = theano.shared(value=initial_vecs, name='init')

        pathvec, updates = theano.map(
                        fn= compop, #compose_TransE,
                        sequences=[idxs_per_path, initial_vecs_fixed],
                        non_sequences=[rels])

        simi = fnsim(pathvec, rel) ### ONLY THIS MAKES SENSE TO DO
        simion = fnsim(pathvec, reln)
        cost, out = margincost(simi, simion, marge)

    if reg != None:
        cost += (reg/2)*relationl.L2_sqr_norm + (reg/2)*relationr.L2_sqr_norm
    
    updates = OrderedDict()
    ### update parameters with a particular learning rate
    params = leftop.params + rightop.params ### KG embedding params
    if hasattr(fnsim, 'params'):
        params += fnsim.params
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
    return theano.function(list_in, [T.mean(cost), T.mean(out)],
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
class Path_model():

    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerTrans()
            self.rightop = Unstructured()

            if state.compop == 'compose_TransE':
                self.compop = compose_TransE
            elif state.compop == 'compose_BilinearDiag':
                self.compop = compose_BilinearDiag
            else:
                raise ValueError("compop is not an acceptable string")
            
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
                self.compop = cPickle.load(f)
                self.leftop = cPickle.load(f)
                self.rightop = cPickle.load(f)
                self.simfn = cPickle.load(f)
                f.close()
            except:
                raise ValueError('could not load model...')

        # function compilation
        self.trainfunc = TrainFnPath(self.compop, state.use_horn_path, \
            self.margincost, self.simfn, self.embeddings, \
            marge=state.marge, reg=state.reg)
        ### even though the training function is for paths, we still 
        ### evaluate on single edges for now...
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
class Bilinear_model():
    '''
        For Bilinear or RESCAL, the score of a triple is just 
        e_h^T W_r e_t (all matrix multiplications) where W_r is a square 
        matrix without any constraints. A separate matrix for each relation. 

        This is implemented as score(e_h, r, e_t) = 
        dot(e_hW_r, e_t)

        where the leftop = e_h^T * W_r 
        the rightop is just identity, but the fnsim between left and right ops
        is constrained to be dot product. We use a special embedding class that
        creates matrices for each relation rather than vectors. 

    '''

    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerBilinear(state.ndim, state.ndim)
            self.rightop = Unstructured()
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                relationsMat = Embeddings(np.random, state.Nrel, \
                    state.ndim * state.ndim, 'relmat') ### must be square,
                ### but unravel the matrix into a vector...
                ### dummy right relations...
                self.embeddings = [embeddings, relationsMat, relationsMat]
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
                raise ValueError('Bilinear must use dotproduct similarity')
            self.simfn = Dotsim 
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('Bilinear must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
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
class DoubleLinear_model():
    '''
        A generalization of Bilinear/RESCAL models: instead of having one 
        matrix betwee two entities, give each entity vector in a triple its 
        own matrix. This is the same as BilinearDiag_extended, except the 
        matrices are not constrained to be diagonal. 

        The score of a triple score(e_h, r, e_t) = 

            fnsim(W_rh * e_h, W_rt * e_t)

        where fnsim is any similarity function defined between two vectors, and
        W_rh and W_rt are square matrices. The leftop and rightops are both
        bilinear, even though this is misleading - the bilinear op actually 
        just takes a matrix and a vector and multiplies them. 
    '''
    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerBilinear(state.ndim, state.ndim)
            self.rightop = LayerBilinear(state.ndim, state.ndim)
            
            # Entity embeddings
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                relationsLeft = Embeddings(np.random, state.Nrel, \
                    state.ndim * state.ndim, 'relmatl') ### must be square,
                relationsRight = Embeddings(np.random, state.Nrel, \
                    state.ndim * state.ndim, 'relmatr')
                ### but unravel the matrix into a vector...
                ### dummy right relations...
                self.embeddings = [embeddings, relationsLeft, relationsRight]
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
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('DoubleLinear must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
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
class STransE_model():
    '''
        From "STransE: a novel embedding mode of entities and relationships in knowledge bases" by Nguyen, Sirts, Qu, and Johnson 2017. 

        The score of a triple is analogous to TransE:
        score(e_h, r, e_t) = ||W_1*h + r - W_2*t||

        make sure to remember that this seeks to MINIMIZE the score for 
        positive triples

        This model is not compositional, so cannot be suited for path queries
    '''
    def __init__(self, state):
        if not state.loadmodel:
            # operators, left and right ops are for distance function
            self.leftop  = LayerSTransE_left(state.ndim, state.ndim)
            self.rightop = LayerSTransE_right(state.ndim, state.ndim)
            
            # Entity embeddings, a modifier matrix for each, and rel vecs
            if not state.loademb:
                embeddings = Embeddings(np.random, state.Nent, state.ndim, \
                    'emb')
                embeddingsModifiersLeft = Embeddings(np.random, state.Nent, \
                    state.ndim * state.ndim, 'emb_mod_left')
                embeddingsModifiersRight = Embeddings(np.random, state.Nent, \
                    state.ndim * state.ndim, 'emb_mod_left')
                relationEmbeddings = Embeddings(np.random, state.Nrel, \
                    state.ndim, 'rels')
                ### but unravel the matrix into a vector...
                ### dummy right relations...
                self.embeddings = [embeddings, embeddingsModifiersLeft, embeddingsModifiersRight, relationEmbeddings]
            else:
                try:
                    state.logger.info('loading embeddings from ' + str(state.loademb))
                    f = open(state.loademb)
                    self.embeddings = cPickle.load(f)
                    f.close()
                except:
                    raise ValueError('could not load embeddings')
        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 4

            ### similarity function of output of left and right ops
            self.simfn = eval(state.simfn + 'sim')  
            if 'pos_high' in state.margincostfunction:
                raise ValueError('STransE_model must use the kind of margin that ranks positive triples lower than negative, e.g. margincost')
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
        self.trainfunc = TrainFn1MemberSTransE(self.margincost, self.simfn, \
            self.embeddings, self.leftop, self.rightop, marge=state.marge, \
            rel=state.rel, reg=state.reg)
        self.ranklfunc = RankLeftFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn, \
            modelType=state.op)
        self.rankrfunc = RankRightFnIdx(self.simfn, self.embeddings, \
            self.leftop, self.rightop, subtensorspec=state.Nsyn, \
            modelType=state.op)
        if state.rel == True: 
            self.rankrelfunc = RankRelFnIdx(self.simfn, self.embeddings, \
                self.leftop, self.rightop, subtensorspec=state.Nsyn_rel, \
                modelType=state.op)
        else:
            self.rankrelfunc = None

# ----------------------------------------------------------------------------
class DNN_model():
    '''
        Use a multilayer perceptron - build up "Operations.Layer" classes. 
    '''
    def __init__(self):
        raise NotImplementedError
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
                self.rightop, marge=state.marge, marg_text=state.marg_text, \
                gamma=state.gamma, rel=state.rel)
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

class BilinearDiag_text_model():
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
                    raise ValueError('could not load model...')

        
            assert type(self.embeddings) is list
            assert len(self.embeddings) == 3


            if state.KBsim != 'Dot':
                raise ValueError('BilinearDiag must use dotproduct similarity')
            ### similarity function of output of left and right ops
            self.KBsim = eval(state.simfn + 'sim') 
            self.textsim = eval(state.textsim + 'sim')
            if 'pos_high' not in state.margincostfunction:
                raise ValueError('BilinearDiag must use the kind of margin that ranks positive triples higher than negative, e.g. margincost_pos_high')
            self.margincost = eval(state.margincostfunction)

            ### Word Embeddings
            self.word_embeddings = WordEmbeddings(state.vocab_size, state.word_dim, wordFile=state.word_file, vocab = state.vocab)

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
