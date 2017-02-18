#! /usr/bin/python
import sys
sys.path.append("../models/")
from KBC_models import *
sys.path.append("../utils/")
from Utils import load_file, convert2idx
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T

def micro_evaluation_statistics(res, n):
    '''
        results is a tuple of (left_ranks, right_ranks) for all test triples
    '''
    left_ranks, right_ranks = res
    dres = {}
    dres['microlmean'] = np.mean(left_ranks)
    dres['microlmrr'] = np.mean(np.reciprocal(left_ranks))        
    dres['microlmedian'] = np.median(left_ranks)
    dres['microlhits@n'] = np.mean(np.asarray(left_ranks) <= n) * 100
    dres['micrormean'] = np.mean(right_ranks)
    dres['micrormrr'] = np.mean(np.reciprocal(right_ranks))                    
    dres['micrormedian'] = np.median(right_ranks)
    dres['microrhits@n'] = np.mean(np.asarray(right_ranks) <= n) * 100
    res_combined = left_ranks + right_ranks
    dres['microgmean'] = np.mean(res_combined)
    dres['microgmedian'] = np.median(res_combined)
    dres['microghits@n'] = np.mean(np.asarray(res_combined) <= n) * 100
    dres['microgmrr'] = np.mean(np.reciprocal(res_combined))
    return dres

def macro_evaluation_statistics(res, idxo, n):
    '''
        computes macro-statistics, which weights mean ranks and hits @ 10 by 
        frequency with which each triple's relationship appears in the data.
    '''
    dres = {}

    listrel = set(idxo)
    dictrelres = {}
    dictrellmean = {}
    dictrelrmean = {}
    dictrelgmean = {}
    dictrellmrr = {}
    dictrelrmrr = {}
    dictrelgmrr = {}
    dictrellmedian = {}
    dictrelrmedian = {}
    dictrelgmedian = {}
    dictrellrn = {}
    dictrelrrn = {}
    dictrelgrn = {}

    for i in listrel:
        dictrelres[i] = [[], []]

    for i, j in enumerate(res[0]):
        dictrelres[idxo[i]][0] += [j]

    for i, j in enumerate(res[1]):
        dictrelres[idxo[i]][1] += [j]

    for i in listrel:
        dictrellmean[i] = np.mean(dictrelres[i][0])
        dictrelrmean[i] = np.mean(dictrelres[i][1])
        dictrelgmean[i] = np.mean(dictrelres[i][0] + dictrelres[i][1])
        dictrellmrr[i] = np.mean(np.reciprocal(dictrelres[i][0]))
        dictrelrmrr[i] = np.mean(np.reciprocal(dictrelres[i][1]))
        dictrelgmrr[i] = np.mean(np.reciprocal(dictrelres[i][0] + dictrelres[i][1]))
        dictrellmedian[i] = np.median(dictrelres[i][0])
        dictrelrmedian[i] = np.median(dictrelres[i][1])
        dictrelgmedian[i] = np.median(dictrelres[i][0] + dictrelres[i][1])
        dictrellrn[i] = np.mean(np.asarray(dictrelres[i][0]) <= n) * 100
        dictrelrrn[i] = np.mean(np.asarray(dictrelres[i][1]) <= n) * 100
        dictrelgrn[i] = np.mean(np.asarray(dictrelres[i][0] +
                                           dictrelres[i][1]) <= n) * 100

    dres['dictrelres']     = dictrelres
    dres['dictrellmean']   = dictrellmean
    dres['dictrelrmean']   = dictrelrmean
    dres['dictrelgmean']   = dictrelgmean
    dres['dictrellmedian'] = dictrellmedian
    dres['dictrelrmedian'] = dictrelrmedian
    dres['dictrelgmedian'] = dictrelgmedian
    dres['dictrellrn']   = dictrellrn
    dres['dictrelrrn']   = dictrelrrn
    dres['dictrelgrn']   = dictrelgrn
    dres['dictrellmrr']  = dictrellmrr
    dres['dictrelrmrr']  = dictrelrmrr
    dres['dictrelgmrr']  = dictrelgmrr    

    dres['macrolmean']   = np.mean(dictrellmean.values())
    dres['macrolmedian'] = np.mean(dictrellmedian.values())
    dres['macrolhits@n'] = np.mean(dictrellrn.values())
    dres['macrormean']   = np.mean(dictrelrmean.values())
    dres['macrormedian'] = np.mean(dictrelrmedian.values())
    dres['macrorhits@n'] = np.mean(dictrelrrn.values())
    dres['macrogmean']   = np.mean(dictrelgmean.values())
    dres['macrogmedian'] = np.mean(dictrelgmedian.values())
    dres['macroghits@n'] = np.mean(dictrelgrn.values())
    dres['macrolmrr']    = np.mean(dictrellmrr.values())
    dres['macrormrr']    = np.mean(dictrelrmrr.values())
    dres['macrogmrr']    = np.mean(dictrelgmrr.values())    

    return dres

def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """
    function to evaluate quality of entity prediction. Used in 
        models/KBC_models.py

    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
 
        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]
    return errl, errr

def RankingScoreIdx(sl, sr, idxl, idxr, idxo):
    """
    To be used only during testing? In RankingEval()

    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: "Score left", Theano function created with RankLeftFnIdx().
    :param sr: "Score Right", Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices, which correspond to test triples
    :param idxr: list of 'right' indices, which correspond to test triples
    :param idxo: list of relation indices, which correspond to test triples
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        errl += [np.argsort(np.argsort((
            sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        errr += [np.argsort(np.argsort((
            sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
        # TODO: err_rel = [np.argsort(np.argsort((
        #    srel(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
        # for some relation scorer srel
    print 'RankingScoreIdx, errl: ' + str(np.shape(errl))
    print 'RankingScoreIdx, errr: ' + str(np.shape(errr))
    return errl, errr

### apparently not used either... but it should be, we should compute
### metrics over lists where multiple valid triples don't appear!
def FilteredRankingScoreIdx(sl, sr, idxl, idxr, idxo, true_triples):
    """
    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    :param sl: Theano function created with RankLeftFnIdx().
    :param sr: Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices.
    :param idxr: list of 'right' indices.
    :param idxo: list of relation indices.
    """
    errl = []
    errr = []
    for l, o, r in zip(idxl, idxo, idxr):
        # print 'new test triple:' + str((l, o, r))
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
 
        ### left 
        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        scores_l = (sl(r, o)[0]).flatten()
        # print '\tremoved ' + str(len(rmv_idx_l)) + ' true triples from left'
        scores_l[rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l)).flatten()[l] + 1]

        ### right
        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]
        scores_r = (sr(l, o)[0]).flatten()
        # print '\tremoved ' + str(len(rmv_idx_r)) + ' true triples from right'
        scores_r[rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r)).flatten()[r] + 1]

    return errl, errr

def RankingEval(datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/',
    dataset='FB15k-test', loadmodel='best_valid_model.pkl', neval='all', 
    Nsyn=14951, n=10, idx2synsetfile='FB15k_idx2entity.pkl'):

    '''
        to be called after training is complete and the best model is saved
        to a file. Then call this file with the path to the best model as
        the first argument
    '''
    print '\nEvaluating model on %s examples from test\nEach example ranked against %s other entities' % (neval, Nsyn)

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    f.close()
    print 'done loading model'

    # Load data
    l = load_file(datapath + dataset + '-lhs.pkl')
    r = load_file(datapath + dataset + '-rhs.pkl')
    o = load_file(datapath + dataset + '-rel.pkl')
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]
    print 'done loading test data'

    # Convert sparse matrix to indexes
    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]
    print 'done converting entities to indices'

    ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn)
    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn)
 
    res = RankingScoreIdx(ranklfunc, rankrfunc, idxl, idxr, idxo)
    print 'RankingEval: ' + str(np.shape(res))
    ### compute micro and macro mean rank and hits @ 10
    dres = micro_evaluation_statistics(res, n)
    dres.update(macro_evaluation_statistics(res, idxo, n))

    print "MICRO RAW:"
    print "\tleft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microlmean'], 3), round(dres['microlmrr'], 3), \
            round(dres['microlmedian'], 3), n, round(dres['microlhits@n'], 3))
    print "\tright mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['micrormean'], 3), round(dres['micrormrr'], 3), \
            round(dres['micrormedian'], 3), n, round(dres['microrhits@n'], 3))
    print "\tglobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microgmean'], 3), round(dres['microgmrr'], 3), \
            round(dres['microgmedian'], 3), n, round(dres['microghits@n'], 3))

    print "MACRO RAW"
    print "\tleft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrolmean'], 3), round(dres['macrolmrr'], 3), \
            round(dres['macrolmedian'], 3), n, round(dres['macrolhits@n'], 3))
    print "\tright mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrormean'], 3), round(dres['macrormrr'], 3), \
            round(dres['macrormedian'], 3), n, round(dres['macrorhits@n'], 3))
    print "\tglobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrogmean'], 3), round(dres['macrogmrr'], 3), \
            round(dres['macrogmedian'], 3), n, round(dres['macroghits@n'], 3))

    return dres

### apparently not used?
def RankingEvalFil(datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/', dataset='FB15k', op='TransE', neval='all',
    loadmodel='best_valid_model.pkl', fold=0, Nrel=14951, Nsyn=1345, n=10):

    '''
        Same as RankingEval, but just excludes any triples that would have 
        been true while iterating through the list of entities for ranking 
        on a particular test triple.

    '''
    print '\nEvaluating model on %s examples from test\nEach example ranked against %s other entities which are FILTERED' % (neval, Nsyn)
    # Load model
    f = open(loadmodel, 'r')
    embeddings = cPickle.load(f)
    leftop =cPickle.load(f)
    rightop =cPickle.load(f)
    simfn = cPickle.load(f) ### wasn't here?
    f.close()
    print 'done loading model'

    # Load data
    l = load_file(datapath + dataset + '-test-lhs.pkl')
    r = load_file(datapath + dataset + '-test-rhs.pkl')
    o = load_file(datapath + dataset + '-test-rel.pkl')
    if type(embeddings) is list:
        o = o[-embeddings[1].N:, :]
    # o = o[-Nrel:, :]
    print 'done loading test triplets'

    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]
    
    tl = load_file(datapath + dataset + '-train-lhs.pkl')
    tr = load_file(datapath + dataset + '-train-rhs.pkl')
    to = load_file(datapath + dataset + '-train-rel.pkl')
    to = to[-Nrel:, :]

    vl = load_file(datapath + dataset + '-valid-lhs.pkl')
    vr = load_file(datapath + dataset + '-valid-rhs.pkl')
    vo = load_file(datapath + dataset + '-valid-rel.pkl')
    vo = vo[-Nrel:, :]
    print 'done loading all true triplets'

    # idxl = convert2idx(l)
    # idxr = convert2idx(r)
    # idxo = convert2idx(o)
    idxtl = convert2idx(tl)
    idxtr = convert2idx(tr)
    idxto = convert2idx(to)
    idxvl = convert2idx(vl)
    idxvr = convert2idx(vr)
    idxvo = convert2idx(vo)
    print 'done converting all data to indices'
    
    # if op == 'Bi':
    #     ranklfunc = RankLeftFnIdxBi(embeddings, leftop, rightop,
    #         subtensorspec=Nsyn)
    #     rankrfunc = RankRightFnIdxBi(embeddings, leftop, rightop,
    #         subtensorspec=Nsyn)
    # elif op == 'Tri':
    #     ranklfunc = RankLeftFnIdxTri(embeddings, leftop, rightop,
    #         subtensorspec=Nsyn)
    #     rankrfunc = RankRightFnIdxTri(embeddings, leftop, rightop,
    #         subtensorspec=Nsyn)
    # elif op == 'TATEC':
    #     ranklfunc = RankLeftFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri,
    #         subtensorspec=Nsyn)
    #     rankrfunc = RankRightFnIdxTATEC(embeddings, leftopbi, leftoptri, rightopbi, rightoptri,
    #         subtensorspec=Nsyn)   
    # else:
    ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
        subtensorspec=Nsyn)
    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
        subtensorspec=Nsyn)     
    
    true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T

    # restest = (error_left entities, error_right_entities)
    restest = FilteredRankingScoreIdx(ranklfunc, rankrfunc, idxl, idxr, idxo, true_triples)

        ### compute micro and macro mean rank and hits @ 10
    dres = micro_evaluation_statistics(restest, n)
    dres.update(macro_evaluation_statistics(restest, idxo, n))

    print "MICRO FILTERED:"
    print "\tleft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microlmean'], 5), round(dres['microlmrr'], 5), \
            round(dres['microlmedian'], 5), n, round(dres['microlhits@n'], 3))
    print "\tright mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['micrormean'], 5), round(dres['micrormrr'], 5), \
            round(dres['micrormedian'], 5),
            n, round(dres['microrhits@n'], 3))
    print "\tglobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microgmean'], 5), round(dres['microgmrr'], 5), \
            round(dres['microgmedian'], 5), n, round(dres['microghits@n'], 3))

    print "MACRO FILTERED"
    print "\tleft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrolmean'], 5), round(dres['macrolmrr'], 5), \
            round(dres['macrolmedian'], 5), n, round(dres['macrolhits@n'], 3))
    print "\tright mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrormean'], 5), round(dres['macrormrr'], 5), \
            round(dres['macrormedian'], 5), n, round(dres['macrorhits@n'], 3))
    print "\tglobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrogmean'], 5), round(dres['macrogmrr'], 5), \
            round(dres['macrogmedian'], 5), n, round(dres['macroghits@n'], 3))

    T10 = np.mean(np.asarray(restest[0] + restest[1]) <= 10) * 100
    MR = np.mean(np.asarray(restest[0] + restest[1]))

    print 'remaining code:'
    print 'hits at 10: ' + str(T10)
    print 'mean rank: ' + str(MR)

    return dres

if __name__ == '__main__':
    '''for instance, call this file from the run/ directory as 
        python evaluate_KBC.py ../run/FB15k_TransE/best_valid_model.pkl
    '''
    # print 'this file is still TODO'
    # exit(1)
    RankingEval(loadmodel=sys.argv[1])
    # RankingEvalFil(loadmodel=sys.argv[1])
