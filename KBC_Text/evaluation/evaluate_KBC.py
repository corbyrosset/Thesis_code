#! /usr/bin/python
import sys
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from KBC_Text.models.KBC_models import *
from KBC_Text.utils.Utils import load_file, convert2idx

def micro_evaluation_statistics(res, n, rel=False):
    '''
        results is a tuple of (left_ranks, right_ranks) for all test triples
    '''
    left_ranks, right_ranks, rel_ranks = res
    left_ranks  = np.array(left_ranks, dtype=np.float64)
    right_ranks = np.array(right_ranks, dtype=np.float64)
    ### TODO: write evaluation for relationship ranking
    dres = {}
    dres['microlmean'] = np.mean(left_ranks)
    dres['microlmrr'] = np.mean(np.reciprocal(left_ranks))        
    dres['microlmedian'] = np.median(left_ranks)
    dres['microlhits@n'] = np.mean((left_ranks) <= n) * 100
    dres['micrormean'] = np.mean(right_ranks)
    dres['micrormrr'] = np.mean(np.reciprocal(right_ranks))                    
    dres['micrormedian'] = np.median(right_ranks)
    dres['microrhits@n'] = np.mean((right_ranks) <= n) * 100
    res_combined = np.append(left_ranks, right_ranks)
    dres['microgmean'] = np.mean(res_combined)
    dres['microgmedian'] = np.median(res_combined)
    dres['microghits@n'] = np.mean((res_combined) <= n) * 100
    dres['microgmrr'] = np.mean(np.reciprocal(res_combined))
    
    if rel:
        rel_ranks = np.array(rel_ranks, dtype=np.float64)
        dres['microrelmean'] = np.mean(rel_ranks)
        dres['microrelmrr'] = np.mean(np.reciprocal(rel_ranks))        
        dres['microrelmedian'] = np.median(rel_ranks)
        dres['microrelhits@n'] = np.mean((rel_ranks) <= n) * 100

    return dres

def macro_evaluation_statistics(res, idxo, n, rel=False):
    '''
        computes macro-statistics, which weights mean ranks and hits @ 10 by 
        frequency with which each triple's relationship appears in the data.
    '''
    dres = {}
    left_ranks, right_ranks, rel_ranks = res

    listrel = set(idxo)
    ranks_per_relation = {}
    left_mean_per_rel = {}
    right_mean_per_rel = {}
    left_mrr_per_rel = {}
    right_mrr_per_rel = {}
    left_med_rank_per_rel = {}
    right_med_rank_per_rel = {}
    left_hitsatn_per_rel = {}
    right_hitsatn_per_rel = {}
    if rel_ranks:
        rel_mean_rank = {}
        rel_mrr = {}
        rel_med_rank = {}
        rel_hitsatn = {}
    gen_mean_rank_per_rel = {}
    gen_mrr_per_rel = {}
    gen_med_rank_per_rel = {}
    gen_hitsatn_per_rel = {}


    for i in listrel:
        ranks_per_relation[i] = [[], [], []] # left, right, relation ranks per relation

    for i, (j, k) in enumerate(zip(left_ranks, right_ranks)):
        assert j > 0
        assert j <= 14951
        assert k > 0
        assert k <= 14951
        # idxo[i] is the relation integer of test triple i
        ranks_per_relation[idxo[i]][0] += [j]
        ranks_per_relation[idxo[i]][1] += [k]

    # for i, j in enumerate(right_ranks):
    #     ranks_per_relation[idxo[i]][1] += [j]
    if rel:
        for i, j in enumerate(rel_ranks):
            ranks_per_relation[idxo[i]][2] += [j]


    for i in listrel: # looks messy, but it's actually simple
        ranks_per_relation[i][0] = np.array(ranks_per_relation[i][0], dtype=np.float64)
        ranks_per_relation[i][1] = np.array(ranks_per_relation[i][1], dtype=np.float64)
        res_combined = np.append(ranks_per_relation[i][0], ranks_per_relation[i][1])
        assert np.size(ranks_per_relation[i][0]) == np.size(ranks_per_relation[i][1])
        assert np.size(res_combined) == 2*np.size(ranks_per_relation[i][1])
        assert np.all(res_combined) > 0
        assert np.all(res_combined) <= 14951

        left_mean_per_rel[i]     = np.mean(ranks_per_relation[i][0])
        right_mean_per_rel[i]    = np.mean(ranks_per_relation[i][1])
        gen_mean_rank_per_rel[i] = np.mean(res_combined)
        left_mrr_per_rel[i] = np.mean(np.reciprocal(ranks_per_relation[i][0]))
        right_mrr_per_rel[i]= np.mean(np.reciprocal(ranks_per_relation[i][1]))
        rec = np.reciprocal(res_combined)
        gen_mrr_per_rel[i]  = np.mean(rec)
        assert np.all(rec > 0)
        assert np.all(rec <= 1)

        # logger.info(np.max(rec), np.min(rec), np.max(ranks_per_relation[i][0]), np.min(ranks_per_relation[i][0]))
        left_med_rank_per_rel[i] = np.median(ranks_per_relation[i][0])
        right_med_rank_per_rel[i]= np.median(ranks_per_relation[i][1])
        gen_med_rank_per_rel[i]  = np.median(res_combined)
        left_hitsatn_per_rel[i]  = np.mean(np.asarray(ranks_per_relation[i][0]) <= n) * 100
        right_hitsatn_per_rel[i] = np.mean(np.asarray(ranks_per_relation[i][1]) <= n) * 100
        gen_hitsatn_per_rel[i]   = np.mean(res_combined <= n) * 100
        
        if rel:
            ranks_per_relation[i][2] = np.array(ranks_per_relation[i][2], dtype=np.float64)
            res_combined = np.append(ranks_per_relation[i][0], ranks_per_relation[i][1])
            res_combined = np.append(res_combined, ranks_per_relation[i][2])
            assert np.all(res_combined) > 0
            assert np.all(res_combined) <= 14951

            rel_mean_rank[i] = np.mean(ranks_per_relation[i][2])
            rel_mrr[i] = np.mean(np.reciprocal(ranks_per_relation[i][2]))
            rel_med_rank[i] = np.median(ranks_per_relation[i][2])
            rel_hitsatn[i] = np.mean(np.asarray(ranks_per_relation[i][2]) <= n) * 100

            # also need to update generalized metrics to include relation ranks
            gen_mean_rank_per_rel[i] = np.mean(res_combined)
            gen_mrr_per_rel[i]       = np.mean(np.reciprocal(res_combined))
            gen_med_rank_per_rel[i]  = np.median(res_combined)
            gen_hitsatn_per_rel[i]   = np.mean(res_combined <= n) * 100


    dres['ranks_per_relation']     = ranks_per_relation
    dres['left_mean_per_rel']      = left_mean_per_rel
    dres['right_mean_per_rel']     = right_mean_per_rel
    dres['gen_mean_rank_per_rel']  = gen_mean_rank_per_rel
    dres['left_med_rank_per_rel']  = left_med_rank_per_rel
    dres['right_med_rank_per_rel'] = right_med_rank_per_rel
    dres['gen_med_rank_per_rel']   = gen_med_rank_per_rel
    dres['left_hitsatn_per_rel']   = left_hitsatn_per_rel
    dres['right_hitsatn_per_rel']  = right_hitsatn_per_rel
    dres['gen_hitsatn_per_rel']    = gen_hitsatn_per_rel
    dres['left_mrr_per_rel']       = left_mrr_per_rel
    dres['right_mrr_per_rel']      = right_mrr_per_rel
    dres['gen_mrr_per_rel']        = gen_mrr_per_rel    
    if rel:
        dres['rel_mean_rank'] = rel_mean_rank
        dres['rel_mrr']       = rel_mrr
        dres['rel_med_rank']  = rel_med_rank
        dres['rel_hitsatn']   = rel_hitsatn

    dres['macrolmean']   = np.mean(left_mean_per_rel.values())
    dres['macrolmedian'] = np.median(left_med_rank_per_rel.values())
    dres['macrolhits@n'] = np.mean(left_hitsatn_per_rel.values())
    dres['macrormean']   = np.mean(right_mean_per_rel.values())
    dres['macrormedian'] = np.median(right_med_rank_per_rel.values())
    dres['macrorhits@n'] = np.mean(right_hitsatn_per_rel.values())
    dres['macrogmean']   = np.mean(gen_mean_rank_per_rel.values())
    dres['macrogmedian'] = np.median(gen_med_rank_per_rel.values())
    dres['macroghits@n'] = np.mean(gen_hitsatn_per_rel.values())
    dres['macrolmrr']    = np.mean(left_mrr_per_rel.values())
    dres['macrormrr']    = np.mean(right_mrr_per_rel.values())
    dres['macrogmrr']    = np.mean(gen_mrr_per_rel.values())  
    if rel:
        dres['macrorelmean']   = np.mean(rel_mean_rank.values())
        dres['macrorelmrr']    = np.mean(rel_mrr.values())
        dres['macrorelmedian'] = np.mean(rel_med_rank.values())
        dres['macrorelhits@n'] = np.mean(rel_hitsatn.values())  

    return dres

def RankingScoreIdx(logger, sl, sr, idxl, idxr, idxo, reverseRanking, rank_rel=None):
    """
    To be used only during testing? In RankingEval()

    This function computes the rank list of the lhs and rhs, over a list of
    lhs, rhs and rel indexes.

    NOTE: if A = np.argsort(np.argsort(a)), then A[i] gives the rank of a[i] in
    a sorted in increasing order. 

    :param sl: "Score left", Theano function created with RankLeftFnIdx().
    :param sr: "Score Right", Theano function created with RankRightFnIdx().
    :param idxl: list of 'left' indices, which correspond to test triples
    :param idxr: list of 'right' indices, which correspond to test triples
    :param idxo: list of relation indices, which correspond to test triples
    """
    errl = []
    errr = []
    err_rel = []
    logger.info('\tRanking: evaluating rank on ' + str(len(idxl)) +' triples')
    ### TODO: these will still be reversed, find a way to un-reverse them!
    for l, o, r in zip(idxl, idxo, idxr):
        ### ORIGINAL:
        # errl += [np.argsort(np.argsort((
        #     sl(r, o)[0]).flatten())[::-1]).flatten()[l] + 1]
        # errr += [np.argsort(np.argsort((
        #     sr(l, o)[0]).flatten())[::-1]).flatten()[r] + 1]
        if reverseRanking: 
        # sorting from high scores to low scores is best to worst
            scores_l = -1*(sl(r, o)[0]).flatten()
            scores_r = -1*(sr(l, o)[0]).flatten()

        else:
        # sorting from low scores to high scores is best worst
            scores_l = (sl(r, o)[0]).flatten()
            scores_r = (sr(l, o)[0]).flatten()

        errl += [np.argsort(np.argsort(scores_l)).flatten()[l] + 1]
        errr += [np.argsort(np.argsort(scores_r)).flatten()[r] + 1]

        if rank_rel is not None:
            # err_rel += [np.argsort(np.argsort((rank_rel(l, r)[0]).flatten())[::-1]).flatten()[o] + 1]
            if reverseRanking:
                scores_rel = -1*(rank_rel(l, r)[0]).flatten()
            else:
                scores_rel = (rank_rel(l, r)[0]).flatten()
            err_rel += [np.argsort(np.argsort(scores_rel)).flatten()[o] + 1]

    if not rank_rel:
        err_rel = [0]*len(idxl)

    return errl, errr, err_rel

### apparently not used either... but it should be, we should compute
### metrics over lists where multiple valid triples don't appear!
def FilteredRankingScoreIdx(logger, sl, sr, idxl, idxr, idxo, true_triples, reverseRanking, rank_rel=None):
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
    err_rel = []
    logger.info('\tFilteredRanking: evaluating rank on ' + str(len(idxl)) +' triples')

    for l, o, r in zip(idxl, idxo, idxr):
        il=np.argwhere(true_triples[:,0]==l).reshape(-1,)
        io=np.argwhere(true_triples[:,1]==o).reshape(-1,)
        ir=np.argwhere(true_triples[:,2]==r).reshape(-1,)
 
        ### left 
        inter_l = [i for i in ir if i in io]
        rmv_idx_l = [true_triples[i,0] for i in inter_l if true_triples[i,0] != l]
        if reverseRanking:
            scores_l = -1*(sl(r, o)[0]).flatten()
            scores_l[rmv_idx_l] = np.inf ### i think this needs to be np.inf
                                         ### rather than -1*np.inf
        else:
            scores_l = (sl(r, o)[0]).flatten()
            scores_l[rmv_idx_l] = np.inf

        # TODO: remove the assert
        ranks = np.argsort(np.argsort(scores_l)).flatten()
        assert (not np.isinf(ranks[l])) and (not np.isinf(-1*ranks[l]))
        errl += [np.argsort(np.argsort(scores_l)).flatten()[l] + 1]

        ### right
        inter_r = [i for i in il if i in io]
        rmv_idx_r = [true_triples[i,2] for i in inter_r if true_triples[i,2] != r]

        if reverseRanking:
            scores_r = -1*(sr(l, o)[0]).flatten()
            scores_r[rmv_idx_r] = np.inf ### i think this needs to be np.inf
        else:
            scores_r = (sr(l, o)[0]).flatten()
            scores_r[rmv_idx_r] = np.inf

        # TODO: remove the assert
        ranks = np.argsort(np.argsort(scores_r)).flatten()
        assert (not np.isinf(ranks[r])) and (not np.isinf(-1*ranks[r]))
        errr += [np.argsort(np.argsort(scores_r)).flatten()[r] + 1]

        ### relations
        if rank_rel is not None:
            inter_o = [i for i in il if i in ir]
            rmv_idx_o = [true_triples[i,1] for i in inter_o if true_triples[i,1] != o]
            ### why is it not this:
            # rmv_idx_o = [i for i in inter_o if true_triples[i,1] != o]
            if reverseRanking:
                scores_rel = -1*(rank_rel(l, r)[0]).flatten()
                scores_rel[rmv_idx_o] = np.inf #i think this needs to be np.inf
            else:
                scores_rel = (rank_rel(l, r)[0]).flatten()
                scores_rel[rmv_idx_o] = np.inf

            # TODO: remove the assert
            ranks = np.argsort(np.argsort(scores_rel)).flatten()
            assert (not np.isinf(ranks[o])) and (not np.isinf(-1*ranks[o]))
            err_rel += [np.argsort(np.argsort(scores_rel)).flatten()[o] + 1]
    
    if not rank_rel:
        err_rel = [0]*len(idxl)

    return errl, errr, err_rel

def RankingEval(datapath, logger, reverseRanking, rel = False, Nsyn_rel = 1345,
    dataset='FB15k-test', loadmodel='best_valid_model.pkl', neval='all', 
    Nsyn=14951, n=10, Nrel = 1345, idx2synsetfile='FB15k_idx2entity.pkl', modelType=None):

    '''
        to be called after training is complete and the best model is saved
        to a file. Then call this file with the path to the best model as
        the first argument
    '''
    logger.info('\tRanking model on %s examples from test. Each example ranked against %s other entities' % (neval, Nsyn))
    if rel == True:
        logger.info('\tRELATION ranking: each rel ranked against %s other rels' %(Nsyn_rel))

    # Load model
    f = open(loadmodel)
    embeddings = cPickle.load(f)
    if modelType and 'Path' in modelType:
        compop = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    if modelType and 'Text' in modelType:
        textsim = cPickle.load(f)
        word_embeddings =cPickle.load(f)
    f.close()

    # Load data
    l = load_file(datapath + dataset + '-lhs.pkl')
    r = load_file(datapath + dataset + '-rhs.pkl')
    o = load_file(datapath + dataset + '-rel.pkl')
    o = o[-Nrel:, :]

    # Convert sparse matrix to indexes
    ### one-dimensional arrays of indices
    if neval == 'all':
        idxl = convert2idx(l)
        idxr = convert2idx(r)
        idxo = convert2idx(o)
    else:
        idxl = convert2idx(l)[:neval]
        idxr = convert2idx(r)[:neval]
        idxo = convert2idx(o)[:neval]

    # idxo -= Nsyn ### I THINK YOU HAVE TO DO THIS TO AVOID INDX OUT OF BOUNDS
    ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn, modelType=modelType)
    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn, modelType=modelType)
    if rel == True:
        rankrelfunc = RankRelFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn_rel, modelType=modelType)
    else:
        rankrelfunc = None
 
    res = RankingScoreIdx(logger, ranklfunc, rankrfunc, idxl, idxr, idxo, reverseRanking, rank_rel=rankrelfunc)
    left_ranks, right_ranks, rel_ranks = res
    logger.info('left_ranks: %s, right_ranks: %s, rel_ranks: %s' % (np.shape(left_ranks), np.shape(right_ranks), np.shape(rel_ranks)))
    logger.info('max left_ranks: %s, max right_ranks: %s, max rel_ranks: %s' % (np.max(left_ranks), np.max(right_ranks), np.max(rel_ranks)))
    
    ### compute micro and macro mean rank and hits @ 10
    dres = micro_evaluation_statistics(res, n, rel)
    dres.update(macro_evaluation_statistics(res, idxo, n, rel))

    logger.info("MICRO RAW:")
    logger.info("\tLeft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microlmean'], 3), round(dres['microlmrr'], 3), \
            round(dres['microlmedian'], 3), n, round(dres['microlhits@n'], 3)))
    logger.info("\tRight mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['micrormean'], 3), round(dres['micrormrr'], 3), \
            round(dres['micrormedian'], 3), n, round(dres['microrhits@n'], 3)))
    if rel == True:
        logger.info("\trelation mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (round(dres['microrelmean'], 3), round(dres['microrelmrr'], \
            3), round(dres['microrelmedian'], 3), n, \
            round(dres['microrelhits@n'], 3)))
    logger.info("\tGlobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microgmean'], 3), round(dres['microgmrr'], 3), \
            round(dres['microgmedian'], 3), n, round(dres['microghits@n'], 3)))

    logger.info("MACRO RAW")
    logger.info("\tLeft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrolmean'], 3), round(dres['macrolmrr'], 3), \
            round(dres['macrolmedian'], 3), n, round(dres['macrolhits@n'], 3)))
    logger.info("\tRight mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrormean'], 3), round(dres['macrormrr'], 3), \
            round(dres['macrormedian'], 3), n, round(dres['macrorhits@n'], 3)))
    if rel == True:
        logger.info("\trelation mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (round(dres['macrorelmean'], 3), round(dres['macrorelmrr'], \
            3), round(dres['macrorelmedian'], 3), n, \
            round(dres['macrorelhits@n'], 3)))
    logger.info("\tGlobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrogmean'], 3), round(dres['macrogmrr'], 3), \
            round(dres['macrogmedian'], 3), n, round(dres['macroghits@n'], 3)))

    return dres

### apparently not used?
def RankingEvalFil(datapath, logger, reverseRanking, rel = False, dataset='FB15k', 
    op='TransE', neval='all', loadmodel='best_valid_model.pkl', 
    fold=0, Nrel=1345, Nsyn=14951, Nsyn_rel = 1345, n=10, modelType=None):

    '''
        Same as RankingEval, but just excludes any triples that would have 
        been true while iterating through the list of entities for ranking 
        on a particular test triple.

    '''
    logger.info('Ranking model on %s examples from test. Each example ranked against %s other entities which are FILTERED' % (neval, Nsyn))
    if rel == True:
        logger.info('RELATION ranking: each rel ranked against %s other rels' %(Nsyn_rel))

    f = open(loadmodel)
    embeddings = cPickle.load(f)
    if modelType and 'Path' in modelType:
        compop = cPickle.load(f)
    leftop = cPickle.load(f)
    rightop = cPickle.load(f)
    simfn = cPickle.load(f)
    if modelType and 'Text' in modelType:
        textsim = cPickle.load(f)
        word_embeddings =cPickle.load(f)
    f.close()

    # Load data
    l = load_file(datapath + dataset + '-test-lhs.pkl')
    r = load_file(datapath + dataset + '-test-rhs.pkl')
    o = load_file(datapath + dataset + '-test-rel.pkl')
    o = o[-Nrel:, :]
    # if type(embeddings) is list:
    #     o = o[-embeddings[1].N:, :]
    # o = o[-Nrel:, :]

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

    idxtl = convert2idx(tl)
    idxtr = convert2idx(tr)
    idxto = convert2idx(to)
    idxvl = convert2idx(vl)
    idxvr = convert2idx(vr)
    idxvo = convert2idx(vo)
    
    ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
        subtensorspec=Nsyn, modelType=modelType)
    rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
        subtensorspec=Nsyn, modelType=modelType) 
    if rel == True:
        rankrelfunc = RankRelFnIdx(simfn, embeddings, leftop, rightop,
            subtensorspec=Nsyn_rel, modelType=modelType)
    else:
        rankrelfunc = None

    true_triples=np.concatenate([idxtl,idxvl,idxl,idxto,idxvo,idxo,idxtr,idxvr,idxr]).reshape(3,idxtl.shape[0]+idxvl.shape[0]+idxl.shape[0]).T

    # restest = (error_left entities, error_right_entities)
    restest = FilteredRankingScoreIdx(logger, ranklfunc, rankrfunc, idxl, idxr, idxo, true_triples, reverseRanking, rank_rel=rankrelfunc)

    ### compute micro and macro mean rank and hits @ 10
    dres = micro_evaluation_statistics(restest, n, rel)
    dres.update(macro_evaluation_statistics(restest, idxo, n, rel))

    logger.info("MICRO FILTERED:")
    logger.info("\tLeft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microlmean'], 5), round(dres['microlmrr'], 5), \
            round(dres['microlmedian'], 5), n, round(dres['microlhits@n'], 3)))
    logger.info("\tRight mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['micrormean'], 5), round(dres['micrormrr'], 5), \
            round(dres['micrormedian'], 5),
            n, round(dres['microrhits@n'], 3)))
    if rel == True:
        logger.info("\trelation mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (round(dres['microrelmean'], 3), round(dres['microrelmrr'], \
            3), round(dres['microrelmedian'], 3), n, \
            round(dres['microrelhits@n'], 3)))
    logger.info("\tGlobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['microgmean'], 5), round(dres['microgmrr'], 5), \
            round(dres['microgmedian'], 5), n, round(dres['microghits@n'], 3)))

    logger.info("MACRO FILTERED")
    logger.info("\tLeft mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrolmean'], 5), round(dres['macrolmrr'], 5), \
            round(dres['macrolmedian'], 5), n, round(dres['macrolhits@n'], 3)))
    logger.info("\tRight mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrormean'], 5), round(dres['macrormrr'], 5), \
            round(dres['macrormedian'], 5), n, round(dres['macrorhits@n'], 3)))
    if rel == True:
        logger.info("\trelation mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (round(dres['macrorelmean'], 3), round(dres['macrorelmrr'], \
            3), round(dres['macrorelmedian'], 3), n, \
            round(dres['macrorelhits@n'], 3)))
    logger.info("\tGlobal mean rank: %s, MRR: %s, median rank: %s, hits@%s: %s%%" % (
            round(dres['macrogmean'], 5), round(dres['macrogmrr'], 5), \
            round(dres['macrogmedian'], 5), n, round(dres['macroghits@n'], 3)))

    return dres

if __name__ == '__main__':
    '''for instance, call this file from the run/ directory as 
        python evaluate_KBC.py ../run/FB15k_TransE/best_valid_model.pkl
    '''
    # RankingEval(loadmodel=sys.argv[1])
    # RankingEvalFil(loadmodel=sys.argv[1])
    logger.info('TODO')
