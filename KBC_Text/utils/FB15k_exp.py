#! /usr/bin/python
import sys
import scipy.sparse as sp
from KBC_Text.utils.Utils import *
from KBC_Text.evaluation.evaluate_KBC import *
from KBC_Text.models.KBC_models import *
from KBC_Text.graph.Graph import Graph


# Experiment function --------------------------------------------------------
def FB15kexp_text(state):
    '''
        Main training loop for text-augmented KG-completion experiment

    '''
    cost_kb, percent_left_kb, percent_rel_kb, percent_right_kb = [], [], [], []
    cost_txt, percent_left_txt, percent_rel_txt, percent_right_txt, percent_txt = [], [], [], [], []
    state.bestvalidMRR = -1
    model = None
    batchsize = -1
    timeref = -1
    reverseRanking = False
    log = state.logger # shorthand

    failedImprovements = 3 # number of chance it gets to improve on Valid MRR
                       # before early stopping

    # Show experiment parameters
    log.info(state)
    np.random.seed(state.seed)

    # Experiment folder
    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    ### load KB data
    trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, \
    validridx, validoidx, testlidx, testridx, testoidx, true_triples, KB \
    = load_FB15k_data(state)
    log.debug(str(np.shape(trainl)) + ' ' + str(np.shape(trainr)) + ' ' + str(np.shape(traino)) + ' ' + str(np.shape(trainlidx)))

    KB_batchsize = trainl.shape[1] / state.nbatches

    ### load TEXT data
    text_trainlidx, text_trainridx, text_trainoidx, text_trainsent, \
    text_validlidx, text_validridx, text_validoidx, text_validsent, \
    text_testlidx, text_testridx, text_testoidx, text_testsent, \
    text_per_triple_cntr, unique_text_per_triple, triple_per_text, \
    sent2idx, idx2sent = load_FB15k_Clueweb_data(state)
    # log.debug(np.shape(text_trainlidx), np.shape(text_trainridx), np.shape(text_trainoidx), np.shape(text_trainsent))
    text_batchsz = text_trainlidx.shape[1] / state.nbatches

    ### get properties of text encoder
    if state.op == 'TransE_text':
        model = TransE_text_model(state)
    elif state.op == 'BilinearDiag_text':
        model = BilinearDiag_model(state)
        reverseRanking = True # don't even give the user the option to set this
        raise NotImplementedError('BilinearDiag_model not yet implemented for textual relations')
    else:
        raise ValueError('Must supply valid model type')
    assert hasattr(model, 'trainFuncKB')
    assert hasattr(model, 'trainFuncText')

    vocab2Idx = model.word_embeddings.getVocab2Idx()
    vocabSize = model.word_embeddings.vocabSize

    if state.rel == True:
        log.debug('Training to rank RELATIONS as well!')
        assert model.rankrelfunc is not None

    log.debug('loaded data and constructed model...')
    log.debug('num epochs: ' + str(state.totepochs))
    log.debug('num batches per epoch: ' + str(state.nbatches))
    log.debug('KB batchsize: ' + str(KB_batchsize))
    log.debug('Textual Triples batchsize: ' + str(text_batchsz))
    log.debug('left and right entity ranking functions will rank a triple against ' + str(state.Nsyn) + ' competitors')

    log.info("BEGIN TRAINING")
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]

        order = np.random.permutation(text_trainlidx.shape[1])
        text_trainlidx[:, order]
        text_trainridx[:, order]
        text_trainoidx[:, order]
        text_trainsent[order]

        ### generate negative samples for KB
        if state.rel == True: 
            trainln, trainon, trainrn = negative_samples_filtered(trainl, traino, trainr, KB, rels=state.Nsyn_rel)
        else:
            trainln, trainrn = negative_samples_filtered(trainl, traino, trainr, KB)
        ### generate negative samples for TEXT data
        text_trainln, text_trainon, text_trainrn = negative_samples_filtered(text_trainlidx, text_trainoidx, text_trainridx, KB, rels=state.Nsyn_rel)

        for i in range(state.nbatches):
            ### Training KB Minibatches
            tmpl = trainl[:, i * KB_batchsize:(i + 1) * KB_batchsize]
            tmpr = trainr[:, i * KB_batchsize:(i + 1) * KB_batchsize]
            tmpo = traino[:, i * KB_batchsize:(i + 1) * KB_batchsize]
            tmpnl = trainln[:, i * KB_batchsize:(i + 1) * KB_batchsize]
            tmpnr = trainrn[:, i * KB_batchsize:(i + 1) * KB_batchsize]
            if state.rel == True:
                tmpno = trainon[:, i * KB_batchsize:(i + 1) * KB_batchsize]

            if state.rel == True: # (has tmpo as additional argument)
                batch_cost, per_l, per_o, per_r = model.trainFuncKB(state.lremb,\
                    state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr, tmpno)
                cost_kb += [batch_cost / float(KB_batchsize)]
                percent_left_kb += [per_l]
                percent_rel_kb += [per_o]
                percent_right_kb += [per_r]

            else:
                batch_cost, per_l, per_r = model.trainFuncKB(state.lremb, \
                    state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr)
                cost_kb += [batch_cost / float(KB_batchsize)]
                percent_left_kb += [per_l]
                percent_right_kb += [per_r]

            ### re-normalize embeddings between KB and TEXT triples?
            if type(model.embeddings) is list:
                model.embeddings[0].normalize()
            else:
                model.embeddings.normalize()

            ### Training Textual Triple Minibatches
            text_tmpl  = text_trainlidx[:, i*text_batchsz:(i + 1)*text_batchsz]
            text_tmpr  = text_trainridx[:, i*text_batchsz:(i + 1)*text_batchsz]
            text_tmpo  = text_trainoidx[:, i*text_batchsz:(i + 1)*text_batchsz]
            text_tmpnl = text_trainln[:, i*text_batchsz:(i + 1)*text_batchsz]
            text_tmpnr = text_trainrn[:, i*text_batchsz:(i + 1)*text_batchsz]
            ### regardles if rel == True, we need the negative relations 4 text
            text_tmpno = text_trainon[:, i*text_batchsz:(i + 1)*text_batchsz]
            sent_idxs  = text_trainsent[i*text_batchsz:(i + 1)*text_batchsz]
            
            # each sentnence is intially just an index into a list of ~ 
            # 6million sentnece, and each sentence is a string of words, some 
            # of which may not be in the word-embedding vocabulary            
            ### TODO: optimize this it's so slow
            ### TODO: investigate how many words are OOV
            sents = [[vocab2Idx.get(i, vocab2Idx['UUUNKKK']) for i in idx2sent[j]] for j in sent_idxs]
            text_tmpsents, inv_lens = binarize_sentences(sents, vocabSize)

            ### TODO: filter out sentences that are too short, or involved in 
            ### too many unique triples
            # keep_indices = filter(lambda x, y: unique_text_per_triple[x] < 6 and y > 2, zip(sent_idx.tolist(), text_sentlen))
            # skip_counter = len(sent_idx) - len(keep_indices)
            # log.debug('\tskipped %s of %s triples bc bad text' % (skip_counter, len(sent_idx)))
            

            ## if using TrainFn1MemberTextONLY!!!
            if state.textual_role == 'TextAsRegularizer':
                batch_cost, per_text = model.trainFuncText(state.lremb, \
                    state.lrparam, text_tmpo, text_tmpno, text_tmpsents, \
                    inv_lens, state.gamma)
                cost_txt += [batch_cost / float(text_batchsz)]
                percent_txt += [per_text]
                percent_left_txt += [0]
                percent_rel_txt += [0]
                percent_right_txt += [0]
            elif state.textual_role == 'TextAsRelation':
                outputs = model.trainFuncText(\
                    state.lremb, state.lrparam, text_tmpl, text_tmpr, \
                    text_tmpnl, text_tmpnr, text_tmpno, \
                    text_tmpsents, inv_lens, state.gamma)
                if state.rel == True:
                    (batch_cost, per_l, per_o, per_r) = outputs
                else:
                    (batch_cost, per_l, per_r) = outputs 
                    per_o = 0
                cost_txt += [batch_cost / float(text_batchsz)]
                percent_txt += [0]
                percent_left_txt += [per_l]
                percent_rel_txt += [per_o]
                percent_right_txt += [per_r]
            elif state.textual_role == 'TextAsRelAndReg':
                outputs = model.trainFuncText(\
                    state.lremb, state.lrparam, text_tmpl, text_tmpr, \
                    text_tmpo, text_tmpnl, text_tmpnr, text_tmpno,\
                    text_tmpsents, inv_lens, state.gamma)
                if state.rel == True:
                    (batch_cost, per_l, per_o, per_r, per_text) = outputs
                else:
                    (batch_cost, per_l, per_r, per_text) = outputs 
                    per_o = 0
                cost_txt += [batch_cost / float(text_batchsz)]
                percent_txt += [per_text]
                percent_left_txt += [per_l]
                percent_rel_txt += [per_o]
                percent_right_txt += [per_r]
            else:
                raise ValueError('Must specify valid textual_role')

            # if state.rel == True:
            #     batch_cost, per_l, per_o, per_r, per_text = model.trainFuncText(state.lremb, state.lrparam, text_tmpl, text_tmpr, text_tmpo, \
            #         text_tmpnl, text_tmpnr, text_tmpno, text_tmpsents, inv_lens, state.gamma)
            #     cost_txt += [batch_cost / float(text_batchsz)]
            #     percent_left_txt += [per_l]
            #     percent_rel_txt += [per_o]
            #     percent_right_txt += [per_r]
            #     percent_txt += [per_text]
            # else:
            #     batch_cost, per_l, per_r, per_text = model.trainFuncText(state.lremb, state.lrparam, text_tmpl, text_tmpr, text_tmpo, \
            #         text_tmpnl, text_tmpnr, text_tmpno, text_tmpsents, inv_lens, state.gamma)
            #     cost_txt += [batch_cost / float(text_batchsz)]
            #     percent_left_txt += [per_l]
            #     percent_right_txt += [per_r]
            #     percent_txt += [per_text]

            if type(model.embeddings) is list:
                model.embeddings[0].normalize()
            else:
                model.embeddings.normalize()

        log.info("----------------------------------------------------------------------------")
        log.info("EPOCH %s (%s seconds):" % (
                epoch_count, round(time.time() - timeref, 3)))
        if state.reg != None:
            log.info("\tAverage L2 norm of relation vector: %s" % \
            (round(np.sqrt(model.embeddings[1].L2_sqr_norm.eval())/ \
            float(state.Nrel), 5)))

        if state.rel:
            log.info("\tCOST KB >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%%" % (\
                round(np.mean(cost_kb), 3), \
                round(np.std(cost_kb), 3), \
                round(np.mean(percent_left_kb) * 100, 2), \
                round(np.mean(percent_rel_kb) * 100, 2), \
                round(np.mean(percent_right_kb) * 100, 2)))
            log.info("\tCOST TEXT >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%% Text: %s%%" % (round(np.mean(cost_txt), 3), \
                round(np.std(cost_txt), 3), \
                round(np.mean(percent_left_txt) * 100, 2), \
                round(np.mean(percent_rel_txt) * 100, 2), \
                round(np.mean(percent_right_txt) * 100, 2), \
                round(np.mean(percent_txt) * 100, 2)))
        else:
            log.info("\tCOST KB >> %s +/- %s, %% updates Left: %s%%  Right: %s%%" % (round(np.mean(cost_kb), 3), \
                round(np.std(cost_kb), 3), \
                round(np.mean(percent_left_kb) * 100, 2),\
                round(np.mean(percent_right_kb) * 100, 2)))
            log.info("\tCOST TEXT >> %s +/- %s, %% updates Left: %s%% Right: %s%% Text: %s%%" % (round(np.mean(cost_txt), 3), \
                round(np.std(cost_txt), 3), \
                round(np.mean(percent_left_txt) * 100, 2), \
                round(np.mean(percent_right_txt) * 100, 2), \
                round(np.mean(percent_txt) * 100, 2)))

        ### reset outputs
        cost_kb, percent_left_kb, percent_rel_kb, percent_right_kb = [], [], [], []
        cost_txt, percent_left_txt, percent_rel_txt, percent_right_txt, percent_txt = [], [], [], [], []
        timeref = time.time()
        
        ### only evaluate on KB triples, not textual ones
        if (epoch_count % state.test_all) == 0:
            ### evaluate by actually computing ranking metrics on some data
            if state.nvalid > 0:
                verl, verr, ver_rel = FilteredRankingScoreIdx(log, model.ranklfunc,\
                    model.rankrfunc, validlidx, validridx, validoidx, \
                    true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                state.validMR = np.mean(verl + verr)# only tune on entity ranking
                state.validMRR = np.mean(np.reciprocal(verl + verr))
            else:
                state.valid = 'not applicable'
            
            if state.ntrain > 0:
                terl, terr, ter_rel = FilteredRankingScoreIdx(log, model.ranklfunc,\
                    model.rankrfunc, trainlidx, trainridx, trainoidx, \
                    true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                state.train = np.mean(terl + terr)# only tune on entity ranking
            else:
                state.train = 'not applicable'
            
            log.info("\tMEAN RANK >> valid: %s, train: %s" % (
                    round(state.validMR, 1), round(state.train, 1)))
            log.info("\t\tMEAN RANK TRAIN: Left: %s Rel: %s Right: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
            log.info("\t\tMEAN RANK VALID: Left: %s Rel: %s Right: %s" % (round(np.mean(verl), 1), round(np.mean(ver_rel), 1), round(np.mean(verr), 1)))
            log.info("\t\tMEAN RECIPROCAL RANK VALID (L & R): %s" % (round(state.validMRR, 4)))

            ### save model that performs best on dev data
            if state.bestvalidMRR == -1 or state.validMRR > state.bestvalidMRR:
                terl, terr, ter_rel = FilteredRankingScoreIdx(log, \
                model.ranklfunc, model.rankrfunc, testlidx, testridx, testoidx, true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                failedImprovements = 3
                state.bestvalidMRR = state.validMRR
                state.besttrain = state.train
                state.besttest = np.mean(terl + terr)
                state.bestepoch = epoch_count
                # Save model best valid model
                f = open(state.savepath + '/best_valid_model.pkl', 'w')
                cPickle.dump(model.embeddings, f, -1)
                cPickle.dump(model.leftop, f, -1)
                cPickle.dump(model.rightop, f, -1)
                cPickle.dump(model.KBsim, f, -1)
                cPickle.dump(model.textsim, f, -1)
                cPickle.dump(model.word_embeddings, f, -1)
                f.close()
                log.info("\tNEW BEST TEST MR: %s" % round(state.besttest, 1))
                log.info("\t\tLeft MR: %s Rel MR: %s Right MR: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
            else:
                if failedImprovements == 0:
                    log.warning("EARLY STOPPING, failed to improve MRR on valid after %s epochs" % (3*state.test_all))
                    break
                    
                log.warning("\tWARNING, failed to improve MRR on valid, %s chances left\n" % (failedImprovements-1))
                failedImprovements -= 1
            
            # Save current model regardless
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(model.embeddings, f, -1)
            cPickle.dump(model.leftop, f, -1)
            cPickle.dump(model.rightop, f, -1)
            cPickle.dump(model.KBsim, f, -1)
            cPickle.dump(model.textsim, f, -1)
            cPickle.dump(model.word_embeddings, f, -1)
            f.close()
            state.nbepochs = epoch_count
            log.info("\t(ranking took %s seconds)" % (
                round(time.time() - timeref, 3)))
            timeref = time.time()
    log.info("----------------------------------------------------------------------------")
    log.info("----------------------------------------------------------------------------")
    return True


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def FB15kexp_path(state):
    '''
        Main training loop

        Early stopping after 3 rankings on validation data with no improvement 
        in MRR. 

        out = list of costs per minibatch for this epoch
        outb = list of percentages of ??? that were updated per minibatch in this epoch??
        restrain & state.train = ranks of all epoch train triples, and the mean thereof
        resvalid & state.valid = ranks of all epoch dev triples, and the mean thereof

    '''
    out, percent_left, percent_rel, percent_right = [], [], [], []
    state.bestvalidMRR = -1
    model = None
    batchsize = -1
    timeref = -1
    reverseRanking = False # for TransE, score the best fit to be near
                           # zero, so ranking is from low to high
                           # in BilinearDiag, it's the opposite
    failedImprovements = 3 # number of chance it gets to improve on Valid MRR
                           # before early stopping
    log = state.logger # shorthand


    # Show experiment parameters
    log.info(state)
    np.random.seed(state.seed)

    # Experiment folder
    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    ### very important: graph has access to entire KB, don't abuse it
    graph = Graph(state, None, text_quads=False)

    ### for negative sampling, yes it's ugly
    _, _, _, trainlidx, trainridx, trainoidx, validlidx, validridx, validoidx, testlidx, testridx, testoidx, true_triples, KB = load_FB15k_data(state)

    ### load data, training data uses paths (trainp) instead of relations
    trainPathRels, devPathRels, testPathRels, trainPathEnts, devPathEnts, \
    testPathEnts, trainPathHeads, devPathHeads, testPathHeads, \
    trainPathTails, devPathTails, testPathTails, trainTargetRels, \
    devTargetRels, testTargetRels = load_FB15k_path_data(state, graph)
    
    log.debug('FB15kexp_path: training paths: %s, dev paths: %s' % (str([i.shape for i in trainPathRels]), str([i.shape for i in devPathRels])))
    ### for instance trainPathRels is [array(1000000, 2), array(1000000, 3)]
    #####################################################################
    train_initial_vecs = np.ones((np.shape(trainPathHeads)[0], state.ndim), dtype=theano.config.floatX) #### FIX THIS IT'S A HACK!!!!
    model = Path_model(state, train_initial_vecs = train_initial_vecs) #trainPathHeads)
    #####################################################################


    assert hasattr(model, 'trainfunc')
    assert hasattr(graph, 'stringToPath')
    reverseRanking = model.reverseRanking

    log.debug('loaded data, constructed model, constructed graph...')
    batchsize = trainPathRels[0].shape[0] / state.nbatches
    log.debug('num epochs: ' + str(state.totepochs))
    log.debug('num batches per epoch: ' + str(state.nbatches))
    log.debug('batchsize: ' + str(batchsize))
    log.debug('left and right entity ranking functions will rank a slot against ' + str(state.Nsyn) + ' competitors')

    if state.loademb and state.nvalid > 0 and state.ntrain > 0:
        log.info("--------------------------------------------------------------------------------------------------")
        log.info('Since we initialized embeddings with pre-trained vectors, establish baseline of their performance:')
        verl, verr, ver_rel = FilteredRankingScoreIdx(log, model.ranklfunc, model.rankrfunc, validlidx, validridx, \
            validoidx, true_triples, reverseRanking)
        validMR = np.mean(verl + verr)# only tune on entity ranking
        validMRR = np.mean(np.reciprocal(verl + verr))

        terl, terr, ter_rel = FilteredRankingScoreIdx(log, \
            model.ranklfunc, model.rankrfunc, trainlidx, \
            trainridx, trainoidx, true_triples, reverseRanking)
        train = np.mean(terl + terr)# only tune on entity ranking
   
        log.info("\tMEAN RANK >> valid: %s, train: %s" % (
                round(validMR, 1), round(train, 1)))
        log.info("\t\tMEAN RANK TRAIN: Left: %s Rel: %s Right: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
        log.info("\t\tMEAN RANK VALID: Left: %s Rel: %s Right: %s" % (round(np.mean(verl), 1), round(np.mean(ver_rel), 1), round(np.mean(verr), 1)))
        log.info("\t\tMEAN RECIPROCAL RANK VALID (L & R): %s" % (round(validMRR, 4)))

    timeref = time.time()
    for length in range(len(trainPathRels)):
        log.info('======================== BEGIN TRAINING: %s ========================' % str(state.graph_files[length]))
        # reset inter-length-dataset statistics
        failedImprovements = 3
        state.bestvalidMRR = -1
        state.besttrain = None
        state.besttest = None
        state.bestepoch = None
        trainPRels = trainPathRels[length]
        devPRels = devPathRels[length]
        testPRels = testPathRels[length]
        trainPEnts = trainPathEnts[length]
        devPEnts = devPathEnts[length]
        testPEnts = testPathEnts[length]
        trainPHeads = expand_to_mat(trainPathHeads[length], state.Nent)
        devPHeads = expand_to_mat(devPathHeads[length], state.Nent)
        testPHeads = expand_to_mat(testPathHeads[length], state.Nent)
        trainPTails = expand_to_mat(trainPathTails[length], state.Nent)
        devPTails = expand_to_mat(devPathTails[length], state.Nent)
        testPTails = expand_to_mat(testPathTails[length], state.Nent)
        trainTrels = expand_to_mat(trainTargetRels[length], state.Nrel) if trainTargetRels[length] else None
        devTrels = expand_to_mat(devTargetRels[length], state.Nrel) if devTargetRels[length] else None
        testTrels = expand_to_mat(testTargetRels[length], state.Nrel) if testTargetRels[length] else None

        # print np.shape(trainPHeads)
        # print np.shape(trainPTails)

        for epoch_count in xrange(1, state.totepochs + 1):
            # Shuffling, note examples are now along 0th axis
            # print (trainPHeads.shape, trainPTails.shape, trainPRels.shape)
            order = np.random.permutation(trainPRels.shape[0])
            ### indices of head, relationpaths, and tail of a path query
            trainl = trainPHeads[:, order]
            trainr = trainPTails[:, order]
            trainPrel = trainPRels[order, :] ### train path relations  
            trainPent = trainPEnts[order, :]
            second_to_last_ent = expand_to_mat(trainPent[:, -1], state.Nent)
            last_rel = expand_to_mat(trainPrel[:, -1], state.Nrel)

            if state.useHornPaths == True: ### needs only negative relations
                traino = trainTrels[:, order]
                trainln, trainon, trainrn = negative_samples_filtered(\
                    second_to_last_ent, traino, trainr, rels=state.Nsyn_rel)
                # print str(np.shape(trainon))
            else: 
                trainln, trainrn = negative_samples_filtered(\
                    trainl, last_rel, trainr, KB)

            for i in range(state.nbatches):
                pathidx = trainPrel[i * batchsize:(i + 1) * batchsize, :]
                if state.useHornPaths: ### needs paths, relations and neg relations
                    tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
                    tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
                    tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
                    tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
                    tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
                    tmpno = trainon[:, i * batchsize:(i + 1) * batchsize]

                    batch_cost, per_l, per_o, per_r = model.trainfunc(\
                        state.lremb, state.lrparam, pathidx, tmpl, \
                        tmpr, tmpnl, tmpnr, tmpo, tmpno)
                    out += [batch_cost / float(batchsize)]
                    percent_left = [per_l] ### TODO!
                    percent_right += [per_r]
                    percent_rel += [per_o]

                else: ### needs only paths and negative right entities

                    tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
                    tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
                    tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
                    tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
                    # print str((np.shape(pathidx), np.shape(tmpl), np.shape(tmpr), np.shape(tmpnr)))
                    batch_cost, per_l, per_r = model.trainfunc(\
                        state.lremb, state.lrparam, pathidx, tmpl, \
                        tmpr, tmpnl, tmpnr)

                    out += [batch_cost / float(batchsize)]
                    percent_left = [per_l]
                    percent_right = [per_r]
                    percent_rel += [0]

                if type(model.embeddings) is list:
                    model.embeddings[0].normalize()
                else:
                    model.embeddings.normalize()

            log.info("----------------------------------------------------------------------------")
            log.info("EPOCH %s (%s seconds):" % (
                    epoch_count, round(time.time() - timeref, 3)))
            if state.reg != None:
                log.info("\tAverage L2 norm of relation vector: %s" % \
                (round(np.sqrt(model.embeddings[1].L2_sqr_norm.eval())/ \
                float(state.Nrel), 5)))
            
            log.info("\tCOST >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%%" % (round(np.mean(out), 3), \
                round(np.std(out), 3), \
                round(np.mean(percent_left) * 100, 2), \
                round(np.mean(percent_rel) * 100, 2), \
                round(np.mean(percent_right) * 100, 2)))

            out, percent_left, percent_rel, percent_right = [], [], [], []
            timeref = time.time()
            
            if (epoch_count % state.test_all) == 0:
                ### evaluate by actually computing ranking metrics on some data
                if state.nvalid > 0:
                    verl, verr, ver_rel = FilteredRankingScoreIdx(log, model.ranklfunc, model.rankrfunc, validlidx, validridx, \
                        validoidx, true_triples, reverseRanking)
                    state.validMR = np.mean(verl + verr)# only tune on entity ranking
                    state.validMRR = np.mean(np.reciprocal(verl + verr))
                else:
                    state.valid = 'not applicable'
                
                if state.ntrain > 0:
                    terl, terr, ter_rel = FilteredRankingScoreIdx(log, \
                        model.ranklfunc, model.rankrfunc, trainlidx, \
                        trainridx, trainoidx, true_triples, reverseRanking)
                    state.train = np.mean(terl + terr)# only tune on entity ranking
                else:
                    state.train = 'not applicable'
                
                log.info("\tMEAN RANK >> valid: %s, train: %s" % (
                        round(state.validMR, 1), round(state.train, 1)))
                log.info("\t\tMEAN RANK TRAIN: Left: %s Rel: %s Right: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
                log.info("\t\tMEAN RANK VALID: Left: %s Rel: %s Right: %s" % (round(np.mean(verl), 1), round(np.mean(ver_rel), 1), round(np.mean(verr), 1)))
                log.info("\t\tMEAN RECIPROCAL RANK VALID (L & R): %s" % (round(state.validMRR, 4)))

                ### save model that performs best on dev data
                if state.bestvalidMRR == -1 or state.validMRR > state.bestvalidMRR:
                    terl, terr, ter_rel = FilteredRankingScoreIdx(log, \
                    model.ranklfunc, model.rankrfunc, testlidx, testridx, testoidx, true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                    
                    failedImprovements = 3
                    state.bestvalidMRR = state.validMRR
                    state.besttrain = state.train
                    state.besttest = np.mean(terl + terr)
                    state.bestepoch = epoch_count
                    # Save model best valid model
                    f = open(state.savepath + '/best_valid_model.pkl', 'w')
                    cPickle.dump(model.embeddings, f, -1)
                    cPickle.dump(model.leftop, f, -1)
                    cPickle.dump(model.rightop, f, -1)
                    cPickle.dump(model.simfn, f, -1)
                    f.close()
                    log.info("\tNEW BEST TEST MR: %s" % round(state.besttest, 1)) 
                    log.info("\t\tLeft MR: %s Rel MR: %s Right MR: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
                else:
                    if failedImprovements == 0:
                        log.warning("EARLY STOPPING, failed to improve MRR on valid after %s epochs" % (3*state.test_all))
                        break
                    log.warning("\tWARNING, failed to improve MRR on valid, %s chances left\n" % (failedImprovements-1))
                    failedImprovements -= 1
                # Save current model regardless
                f = open(state.savepath + '/current_model.pkl', 'w')
                cPickle.dump(model.embeddings, f, -1)
                cPickle.dump(model.compop, f, -1)
                cPickle.dump(model.leftop, f, -1)
                cPickle.dump(model.rightop, f, -1)
                cPickle.dump(model.simfn, f, -1)
                f.close()
                state.nbepochs = epoch_count
                log.info("\t(ranking took %s seconds)" % (
                    round(time.time() - timeref, 3)))
                timeref = time.time()


    log.info("----------------------------------------------------------------------------")
    log.info("----------------------------------------------------------------------------")
    return True

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def FB15kexp(state):
    '''
        Main training loop

        Early stopping after 3 rankings on validation data with no improvement 
        in MRR. 

        out = list of costs per minibatch for this epoch
        outb = list of percentages of ??? that were updated per minibatch in this epoch??
        restrain & state.train = ranks of all epoch train triples, and the mean thereof
        resvalid & state.valid = ranks of all epoch dev triples, and the mean thereof

    '''
    out, percent_left, percent_rel, percent_right = [], [], [], []
    state.bestvalidMRR = -1
    model = None
    batchsize = -1
    timeref = -1
    reverseRanking = False # for TransE, score the best fit to be near
                           # zero, so ranking is from low to high
                           # in BilinearDiag, it's the opposite
    failedImprovements = 3 # number of chance it gets to improve on Valid MRR
                           # before early stopping
    log = state.logger # shorthand


    # Show experiment parameters
    log.info(state)
    np.random.seed(state.seed)

    # Experiment folder
    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    ### load data
    trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, \
    validridx, validoidx, testlidx, testridx, testoidx, true_triples, KB \
    = load_FB15k_data(state)
    log.debug('FB15kexp: load_FB15k_data(): trainl%s, trainr%s, traino%s, trainlidx%s' % (np.shape(trainl), np.shape(trainr), np.shape(traino), np.shape(trainlidx)))

    ### model has properties: trainfunc, ranklfunc, rankrfunc, 
    ### embeddings, leftop, rightop, and simfn
    if state.op == 'TransE':
        model = TransE_model(state)
    elif state.op == 'STransE':
        model = STransE_model(state)
    elif state.op == 'BilinearDiag':
        model = BilinearDiag_model(state)
        reverseRanking = True
    elif state.op == 'ModelE':
        model = ModelE_model(state)
        reverseRanking = True
    elif state.op == 'BilinearDiagExtended':
        model = BilinearDiagExtended_model(state)
        reverseRanking = state.reverseRanking # could be either
    elif state.op == 'Baseline1':
        model = Baseline1_model(state)
        reverseRanking = True
    elif state.op == 'Bilinear':
        model = Bilinear_model(state)
        reverseRanking = True
    elif state.op == 'DoubleLinear':
        model = DoubleLinear_model(state)
        reverseRanking = state.reverseRanking
    else:
        raise ValueError('Must supply valid model type')
    assert hasattr(model, 'trainfunc')

    if state.rel == True:
        log.debug('Training to rank RELATIONS as well!')
        assert model.rankrelfunc is not None

    log.debug('loaded data and constructed model...')
    batchsize = trainl.shape[1] / state.nbatches
    log.debug('num epochs: ' + str(state.totepochs))
    log.debug('num batches per epoch: ' + str(state.nbatches))
    log.debug('batchsize: ' + str(batchsize))
    log.debug('left and right entity ranking functions will rank a slot against ' + str(state.Nsyn) + ' competitors')

    log.info("BEGIN TRAINING")
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]

        if state.rel == True:
            trainln, trainon, trainrn = negative_samples_filtered(trainl, traino, trainr, KB, rels=state.Nsyn_rel)
        else:
            trainln, trainrn = negative_samples_filtered(trainl, traino, trainr, KB)

        for i in range(state.nbatches):
            tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
            tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
            tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
            tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
            if state.rel == True:
                tmpno = trainon[:, i * batchsize:(i + 1) * batchsize]
            
            # training iteration
            if state.rel == True: # (has tmpo as additional argument)
                batch_cost, per_l, per_o, per_r = model.trainfunc(state.lremb,\
                    state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr, tmpno)
                out += [batch_cost / float(batchsize)]
                percent_left += [per_l]
                percent_rel += [per_o]
                percent_right += [per_r]

            else:
                batch_cost, per_l, per_r = model.trainfunc(state.lremb, \
                    state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr)
                out += [batch_cost / float(batchsize)]
                percent_left += [per_l]
                percent_right += [per_r]

            # embeddings normalization
            ### TODO: why only normalize embeddings[0]?? only normalize entity 
            ### embs, not the relationships ones??
            if type(model.embeddings) is list:
                model.embeddings[0].normalize()
            else:
                model.embeddings.normalize()

        log.info("----------------------------------------------------------------------------")
        log.info("EPOCH %s (%s seconds):" % (
                epoch_count, round(time.time() - timeref, 3)))
        if state.reg != None:
            log.info("\tAverage L2 norm of relation vector: %s" % \
            (round(np.sqrt(model.embeddings[1].L2_sqr_norm.eval())/ \
            float(state.Nrel), 5)))
        if state.rel:
            log.info("\tCOST >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%%" % (round(np.mean(out), 3), \
                round(np.std(out), 3), \
                round(np.mean(percent_left) * 100, 2), \
                round(np.mean(percent_rel) * 100, 2), \
                round(np.mean(percent_right) * 100, 2)))
        else:
            log.info("\tCOST >> %s +/- %s, %% updates Left: %s%%  Right: %s%%" % (round(np.mean(out), 3),round(np.std(out), 3),\
                round(np.mean(percent_left) * 100, 2), \
                round(np.mean(percent_right) * 100, 2)))

        out, percent_left, percent_rel, percent_right = [], [], [], []
        timeref = time.time()
        
        if (epoch_count % state.test_all) == 0:
            ### evaluate by actually computing ranking metrics on some data
            if state.nvalid > 0:
                verl, verr, ver_rel = FilteredRankingScoreIdx(log, model.ranklfunc,\
                    model.rankrfunc, validlidx, validridx, validoidx, \
                    true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                state.validMR = np.mean(verl + verr)# only tune on entity ranking
                state.validMRR = np.mean(np.reciprocal(verl + verr))
            else:
                state.valid = 'not applicable'
            
            if state.ntrain > 0:
                terl, terr, ter_rel = FilteredRankingScoreIdx(log, model.ranklfunc,\
                    model.rankrfunc, trainlidx, trainridx, trainoidx, \
                    true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                state.train = np.mean(terl + terr)# only tune on entity ranking
            else:
                state.train = 'not applicable'
            
            log.info("\tMEAN RANK >> valid: %s, train: %s" % (
                    round(state.validMR, 1), round(state.train, 1)))
            log.info("\t\tMEAN RANK TRAIN: Left: %s Rel: %s Right: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
            log.info("\t\tMEAN RANK VALID: Left: %s Rel: %s Right: %s" % (round(np.mean(verl), 1), round(np.mean(ver_rel), 1), round(np.mean(verr), 1)))
            log.info("\t\tMEAN RECIPROCAL RANK VALID (L & R): %s" % (round(state.validMRR, 4)))

            ### save model that performs best on dev data
            if state.bestvalidMRR == -1 or state.validMRR > state.bestvalidMRR:
                terl, terr, ter_rel = FilteredRankingScoreIdx(log, \
                model.ranklfunc, model.rankrfunc, testlidx, testridx, testoidx, true_triples, reverseRanking, rank_rel=model.rankrelfunc)
                
                failedImprovements = 3
                state.bestvalidMRR = state.validMRR
                state.besttrain = state.train
                state.besttest = np.mean(terl + terr)
                state.bestepoch = epoch_count
                # Save model best valid model
                f = open(state.savepath + '/best_valid_model.pkl', 'w')
                cPickle.dump(model.embeddings, f, -1)
                cPickle.dump(model.leftop, f, -1)
                cPickle.dump(model.rightop, f, -1)
                cPickle.dump(model.simfn, f, -1)
                f.close()
                log.info("\tNEW BEST TEST MR: %s" % round(state.besttest, 1)) 
                log.info("\t\tLeft MR: %s Rel MR: %s Right MR: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1)))
            else:
                if failedImprovements == 0:
                    log.warning("EARLY STOPPING, failed to improve MRR on valid after %s epochs" % (3*state.test_all))
                    break
                log.warning("\tWARNING, failed to improve MRR on valid, %s chances left\n" % (failedImprovements-1))
                failedImprovements -= 1
            # Save current model regardless
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(model.embeddings, f, -1)
            cPickle.dump(model.leftop, f, -1)
            cPickle.dump(model.rightop, f, -1)
            cPickle.dump(model.simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            log.info("\t(ranking took %s seconds)" % (
                round(time.time() - timeref, 3)))
            timeref = time.time()


    log.info("----------------------------------------------------------------------------")
    log.info("----------------------------------------------------------------------------")
    return True

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def launch(identifier, experiment_type, logger, datapath='data/', \
    dataset='FB15k', \
    Nent=14951, rhoE=1, margincostfunction='margincost', reg=0.01, \
    rhoL=5, Nsyn_rel = 1345, Nsyn=14951, Nrel=1345, loadmodel=False, \
    loademb=False, op='Unstructured', simfn='Dot', ndim=50, marge=1., \
    lremb=0.1, lrparam=1., nbatches=100, totepochs=2000, test_all=1, \
    ntrain = 'all', nvalid = 'all', ntest = 'all', seed=123, \
    savepath='', rel=False, reverseRanking=False):

    # Argument of the experiment script
    state = DD()
    state.identifier = identifier
    state.datapath = datapath
    state.dataset = dataset
    state.savepath = savepath
    state.experiment_type = experiment_type
    state.margincostfunction = margincostfunction
    state.reg = reg


    state.Nent = Nent # Total number of entities
    state.Nsyn = Nsyn # number of entities against which to rank a given test
                      # set entity. It could be all entities (14951) or less,
                      # or it could be filtered in a special way.
    state.Nrel = Nrel # number of relations
    state.Nsyn_rel = Nsyn_rel # number of relatins against which to rank a 
                              # given missing relation
    state.loadmodel = loadmodel
    # state.loadmodelBi = loadmodelBi
    # state.loadmodelTri = loadmodelTri
    state.loademb = loademb ### load previously trained embeddings?
    
    state.op = op
    state.simfn = simfn
    state.ndim = ndim # dimension of both relationship and entity embeddings
    state.marge = marge # margin used in ranking loss
    state.reverseRanking = reverseRanking
    # state.rhoE = rhoE
    # state.rhoL = rhoL
    state.lremb = lremb     # learning rate for embeddings
    state.lrparam = lrparam # learning rate for params of leftop, rightop, 
                            # and fnsim, if they have parametesr
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all # when training, how many epochs until use 
                              # validation set again
    state.rel = rel # whether to also train a relationship ranker in TrainFunc
    state.ntrain = ntrain # num exs from train to compute a rank for, 
                          # 'all' or some num
    state.nvalid = nvalid # num exs from valid to compute a rank for
    state.ntest = ntest   # num exs from test to compute a rank for
    state.seed = seed

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    state.logger = logger
    FB15kexp(state)

def launch_path(identifier, experiment_type, logger, datapath='data/', \
    dataset='FB15k', useHornPaths=False, compop='compose_TransE', \
    Nent=14951, rhoE=1, margincostfunction='margincost', reg=0.01, \
    rhoL=5, Nsyn_rel = 1345, Nsyn=14951, Nrel=1345, loadmodel=False, \
    loademb=False, simfn='Dot', ndim=50, marge=1., \
    lremb=0.1, lrparam=1., nbatches=100, totepochs=500, test_all=1, \
    ntrain = 'all', nvalid = 'all', ntest = 'all', seed=123, \
    savepath='', rel=False, needIntermediateNodesOnPaths=False, graph_files=['length_2_numPaths_50000000']):

    # Argument of the experiment script
    state = DD()
    state.identifier = identifier
    state.datapath = datapath
    state.dataset = dataset
    state.savepath = savepath
    state.experiment_type = experiment_type
    
    state.margincostfunction = margincostfunction
    state.reg = reg
    state.Nent = Nent # Total number of entities
    state.Nsyn = Nsyn # number of entities against which to rank a given test
                      # set entity. It could be all entities (14951) or less,
                      # or it could be filtered in a special way.
    state.Nrel = Nrel # number of relations
    state.Nsyn_rel = Nsyn_rel # number of relatins against which to rank a 
                              # given missing relation
    state.loadmodel = loadmodel
    state.loademb = loademb # load previously trained embeddings
    state.simfn = simfn
    state.ndim = ndim       # dimension of relationship and entity embeddings
    state.marge = marge     # margin used in ranking loss
    state.lremb = lremb     # learning rate for embeddings
    state.lrparam = lrparam # learning rate for params of leftop, rightop, 
                            # and fnsim, if they have parametesr
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all # when training, how many epochs until use 
                              # validation set again
    state.rel = rel # whether to also train a relationship ranker in TrainFunc
    state.ntrain = ntrain # num exs from train to compute a rank for, 
                          # 'all' or some num
    state.nvalid = nvalid # num exs from valid to compute a rank for
    state.ntest = ntest   # num exs from test to compute a rank for
    state.seed = seed

    state.graph_files = graph_files
    state.compop = compop
    state.useHornPaths = useHornPaths
    state.needIntermediateNodesOnPaths = needIntermediateNodesOnPaths

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    state.logger = logger
    FB15kexp_path(state)


def launch_text(identifier, experiment_type, logger, datapath='data/', \
    dataset='FB15k', Nent=16296, Nsyn_rel = 1345, reg=None, \
    Nsyn=14951, Nrel=1345, loadmodel=False, margincostfunction='margincost', \
    loademb=False, op='Unstructured', simfn='Dot', ndim=50, marge=1., \
    lremb=0.1, lrparam=1., nbatches=100, totepochs=2000, test_all=1, \
    ntrain = 'all', nvalid = 'all', ntest = 'all', textsim = 'L2', \
    vocab_size = 100000, word_dim = 300, word_file = None, vocab = None, \
    gamma = 0.01, seed=123, savepath='', rel=False, numTextTrain=10000, \
    marg_text=1.0, textual_role='TextAsRegularizer'):

    # Argument of the experiment script
    state = DD()
    state.identifier = identifier
    state.datapath = datapath
    state.dataset = dataset
    state.savepath = savepath
    state.experiment_type = experiment_type
    state.margincostfunction = margincostfunction


    state.Nent = Nent # Total number of entities
    state.Nsyn = Nsyn # number of entities against which to rank a given test
                      # set entity. It could be all entities (14951) or less,
                      # or it could be filtered in a special way.
    state.Nrel = Nrel # number of relations
    state.Nsyn_rel = Nsyn_rel # number of relatins against which to rank a 
                              # given missing relation
    state.loadmodel = loadmodel
    # state.loadmodelBi = loadmodelBi
    # state.loadmodelTri = loadmodelTri
    state.loademb = loademb ### load previously trained embeddings?
    
    state.op = op
    state.simfn = simfn
    state.ndim = ndim # dimension of both relationship and entity embeddings
    state.marge = marge # margin used in ranking loss
    # state.rhoE = rhoE
    # state.rhoL = rhoL
    state.lremb = lremb     # learning rate for embeddings
    state.lrparam = lrparam # learning rate for params of leftop, rightop, 
                            # and fnsim, if they have parametesr

    state.textual_role = textual_role # whether to interpret a text embedding
                                      # as a form or regularization or as a 
                                      # relation
    state.marg_text = marg_text
    state.textsim = textsim  # how to compare a textual relation to KB relation
    state.vocab_size = vocab_size # size of vocabulary
    state.vocab = vocab # a hashset of vocab words
    state.word_dim = word_dim # dimension of each word embedding
    state.word_file = word_file # path to file containing word embeddings
    state.gamma = gamma # weight to use for cost of textual triple
    state.numTextTrain = numTextTrain

    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all # when training, how many epochs until use 
                              # validation set again
    state.rel = rel # whether to also train a relationship ranker in TrainFunc
    state.reg = reg
    state.ntrain = ntrain # num exs from train to compute a rank for, 
                          # 'all' or some num
    state.nvalid = nvalid # num exs from valid to compute a rank for
    state.ntest = ntest   # num exs from test to compute a rank for
    state.seed = seed

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    state.logger = logger
    FB15kexp_text(state)


if __name__ == '__main__':
    launch()
