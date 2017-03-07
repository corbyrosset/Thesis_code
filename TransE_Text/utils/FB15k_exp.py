#! /usr/bin/python
import sys
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/models/")
from KBC_models import *
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/evaluation/")
from evaluate_KBC import *
from Utils import * 

# Jobman channel remplacement
class Channel(object):
    def __init__(self, state):
        self.state = state
        f = open(self.state.savepath + '/orig_state.pkl', 'w')
        cPickle.dump(self.state, f, -1)
        f.close()
        self.COMPLETE = 1

    def save(self):
        f = open(self.state.savepath + '/current_state.pkl', 'w')
        cPickle.dump(self.state, f, -1)
        f.close()

# Experiment function --------------------------------------------------------
def FB15kexp_text(state, channel):
    '''
        Main training loop for text-augmented KG-completion experiment

    '''
    cost_kb, percent_left_kb, percent_rel_kb, percent_right_kb = [], [], [], []
    cost_txt, percent_left_txt, percent_rel_txt, percent_right_txt, percent_txt = [], [], [], [], []
    state.bestvalid = -1
    model = None
    batchsize = -1
    timeref = -1

    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)

    # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)

    ### load KB data
    trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, \
    validridx, validoidx, testlidx, testridx, testoidx, true_triples, KB \
    = load_FB15k_data(state)
    print np.shape(trainl), np.shape(trainr), np.shape(traino), np.shape(trainlidx)

    KB_batchsize = trainl.shape[1] / state.nbatches

    ### load TEXT data
    text_trainlidx, text_trainridx, text_trainoidx, text_trainsent, \
    text_validlidx, text_validridx, text_validoidx, text_validsent, \
    text_testlidx, text_testridx, text_testoidx, text_testsent, \
    text_per_triple_cntr, unique_text_per_triple, triple_per_text, \
    sent2idx, idx2sent = load_FB15k_Clueweb_data(state)
    print np.shape(text_trainlidx), np.shape(text_trainridx), np.shape(text_trainoidx), np.shape(text_trainsent)
    text_batchsz = text_trainlidx.shape[1] / state.nbatches

    ### get properties of text encoder
    model = TransE_text_model(state)
    vocab2Idx = model.word_embeddings.getVocab2Idx()
    vocabSize = model.word_embeddings.vocabSize

    if state.rel == True:
        print 'Training to rank RELATIONS as well!'
        assert model.rankrelfunc is not None

    print 'loaded data and constructed model...'
    print 'num epochs: ' + str(state.totepochs)
    print 'num batches per epoch: ' + str(state.nbatches)
    print 'KB batchsize: ' + str(KB_batchsize)
    print 'Textual Triples batchsize: ' + str(text_batchsz)
    print 'left and right entity ranking functions will rank a triple against ' + str(state.Nsyn) + ' competitors'

    print >> sys.stderr, "BEGIN TRAINING"
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
                cost_kb += [batch_cost / float(batchsize)]
                percent_left_kb += [per_l]
                percent_rel_kb += [per_o]
                percent_right_kb += [per_r]

            else:
                batch_cost, per_l, per_r = model.trainFuncKB(state.lremb, \
                    state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr)
                cost_kb += [batch_cost / float(batchsize)]
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
            # print '\tskipped %s of %s triples bc bad text' % (skip_counter, len(sent_idx))
            

            ### if using TrainFn1MemberTextONLY!!!
            # batch_cost, per_text = model.trainFuncText(state.lremb, \
            #     state.lrparam, text_tmpo, text_tmpno, text_tmpsents, \
            #     inv_lens, state.gamma)
            # cost_txt += [batch_cost / float(text_batchsz)]
            # percent_txt += [per_text]
            
            if state.rel == True:
                batch_cost, per_l, per_o, per_r, per_text = model.trainFuncText(state.lremb, state.lrparam, text_tmpl, text_tmpr, text_tmpo, \
                    text_tmpnl, text_tmpnr, text_tmpno, text_tmpsents, inv_lens, state.gamma)
                cost_txt += [batch_cost / float(batchsize)]
                percent_left_txt += [per_l]
                percent_rel_txt += [per_o]
                percent_right_txt += [per_r]
                percent_txt += [per_text]
            else:
                batch_cost, per_l, per_r, per_text = model.trainFuncText(state.lremb, state.lrparam, text_tmpl, text_tmpr, text_tmpo, \
                    text_tmpnl, text_tmpnr, text_tmpno, text_tmpsents, inv_lens, state.gamma)
                cost_txt += [batch_cost / float(batchsize)]
                percent_left_txt += [per_l]
                percent_right_txt += [per_r]
                percent_txt += [per_text]

            if type(model.embeddings) is list:
                model.embeddings[0].normalize()
            else:
                model.embeddings.normalize()

        print >> sys.stderr, "------------------------------------------------------------"
        print >> sys.stderr, "EPOCH %s (%s seconds):" % (
                epoch_count, round(time.time() - timeref, 3))

        if state.rel:
            print >> sys.stderr, "\tCOST KB >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%%" % (\
                round(np.mean(cost_kb), 3), \
                round(np.std(cost_kb), 3), \
                round(np.mean(percent_left_kb) * 100, 2), \
                round(np.mean(percent_rel_kb) * 100, 2), \
                round(np.mean(percent_right_kb) * 100, 2))
            print >> sys.stderr, "\tCOST TEXT >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%% Text: %s%%" % (round(np.mean(cost_txt), 3), \
                round(np.std(cost_txt), 3), \
                round(np.mean(percent_left_txt) * 100, 2), \
                round(np.mean(percent_rel_txt) * 100, 2), \
                round(np.mean(percent_right_txt) * 100, 2), \
                round(np.mean(percent_txt) * 100, 2))
        else:
            print >> sys.stderr, "\tCOST KB >> %s +/- %s, %% updates Left: %s%%  Right: %s%%" % (round(np.mean(cost_kb), 3), \
                round(np.std(cost_kb), 3), \
                round(np.mean(percent_left_kb) * 100, 2),\
                round(np.mean(percent_right_kb) * 100, 2))
            print >> sys.stderr, "\tCOST TEXT >> %s +/- %s, %% updates Left: %s%% Right: %s%% Text: %s%%" % (round(np.mean(cost_txt), 3), \
                round(np.std(cost_txt), 3), \
                round(np.mean(percent_left_txt) * 100, 2), \
                round(np.mean(percent_right_txt) * 100, 2), \
                round(np.mean(percent_txt) * 100, 2))

        ### reset outputs
        cost_kb, percent_left_kb, percent_rel_kb, percent_right_kb = [], [], [], []
        cost_txt, percent_left_txt, percent_rel_txt, percent_right_txt, percent_txt = [], [], [], [], []
        timeref = time.time()
        
        ### only evaluate on KB triples, not textual ones
        if (epoch_count % state.test_all) == 0:
            ### evaluate by actually computing ranking metrics on some data
            if state.nvalid > 0:
                verl, verr, ver_rel = FilteredRankingScoreIdx(model.ranklfunc,\
                    model.rankrfunc, validlidx, validridx, validoidx, \
                    true_triples, rank_rel=model.rankrelfunc)
                state.valid = np.mean(verl + verr)# only tune on entity ranking
            else:
                state.valid = 'not applicable'
            
            if state.ntrain > 0:
                terl, terr, ter_rel = FilteredRankingScoreIdx(model.ranklfunc,\
                    model.rankrfunc, trainlidx, trainridx, trainoidx, \
                    true_triples, rank_rel=model.rankrelfunc)
                state.train = np.mean(terl + terr)# only tune on entity ranking
            else:
                state.train = 'not applicable'
            
            print >> sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (
                    round(state.valid, 1), round(state.train, 1))
            print >> sys.stderr, "\t\tMEAN RANK TRAIN: Left: %s Rel: %s Right: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1))
            print >> sys.stderr, "\t\tMEAN RANK VALID: Left: %s Rel: %s Right: %s" % (round(np.mean(verl), 1), round(np.mean(ver_rel), 1), round(np.mean(verr), 1))

            ### save model that performs best on dev data
            if state.bestvalid == -1 or state.valid < state.bestvalid:
                terl, terr, ter_rel = FilteredRankingScoreIdx(model.ranklfunc,\
                    model.rankrfunc, testlidx, testridx, testoidx,true_triples)
                state.bestvalid = state.valid
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
                print >> sys.stderr, "\tNEW BEST TEST: %s\n\t\tMEAN RANK TEST Left: %s Rel: %s Right: %s" % (round(state.besttest, 1), round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1))
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(model.embeddings, f, -1)
            cPickle.dump(model.leftop, f, -1)
            cPickle.dump(model.rightop, f, -1)
            cPickle.dump(model.KBsim, f, -1)
            cPickle.dump(model.textsim, f, -1)
            cPickle.dump(model.word_embeddings, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(ranking took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()
    print >> sys.stderr, "------------------------------------------------------------"
    print >> sys.stderr, "------------------------------------------------------------"
    return channel.COMPLETE


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def FB15kexp(state, channel):
    '''
        Main training loop

        out = list of costs per minibatch for this epoch
        outb = list of percentages of ??? that were updated per minibatch in this epoch??
        restrain & state.train = ranks of all epoch train triples, and the mean thereof
        resvalid & state.valid = ranks of all epoch dev triples, and the mean thereof

    '''
    out, percent_left, percent_rel, percent_right = [], [], [], []
    state.bestvalid = -1
    model = None
    batchsize = -1
    timeref = -1

    # Show experiment parameters
    print >> sys.stderr, state
    np.random.seed(state.seed)

    # Experiment folder
    if hasattr(channel, 'remote_path'):
        state.savepath = channel.remote_path + '/'
    elif hasattr(channel, 'path'):
        state.savepath = channel.path + '/'
    else:
        if not os.path.isdir(state.savepath):
            os.mkdir(state.savepath)

    ### load data
    trainl, trainr, traino, trainlidx, trainridx, trainoidx, validlidx, \
    validridx, validoidx, testlidx, testridx, testoidx, true_triples, KB \
    = load_FB15k_data(state)
    print np.shape(trainl), np.shape(trainr), np.shape(traino), np.shape(trainlidx)

    ### model has properties: trainfunc, ranklfunc, rankrfunc, 
    ### embeddings, leftop, rightop, and simfn
    model = TransE_model(state)
    if state.rel == True:
        print 'Training to rank RELATIONS as well!'
        assert model.rankrelfunc is not None

    print 'loaded data and constructed model...'
    batchsize = trainl.shape[1] / state.nbatches
    print 'num epochs: ' + str(state.totepochs)
    print 'num batches per epoch: ' + str(state.nbatches)
    print 'batchsize: ' + str(batchsize)
    print 'left and right entity ranking functions will rank a slot against ' + str(state.Nsyn) + ' competitors'

    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        
        # Negatives, TODO, these should be filtered as well?
        # trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
        # trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))
        # if state.rel == True: ### create negative relationship instances
        #     trainon = create_random_mat(traino.shape, np.arange(state.Nsyn_rel))

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

        print >> sys.stderr, "------------------------------------------------------"
        print >> sys.stderr, "EPOCH %s (%s seconds):" % (
                epoch_count, round(time.time() - timeref, 3))
        if state.rel:
            print >> sys.stderr, "\tCOST >> %s +/- %s, %% updates Left: %s%% Rel: %s%% Right: %s%%" % (round(np.mean(out), 3), \
                round(np.std(out), 3), \
                round(np.mean(percent_left) * 100, 2), \
                round(np.mean(percent_rel) * 100, 2), \
                round(np.mean(percent_right) * 100, 2))
        else:
            print >> sys.stderr, "\tCOST >> %s +/- %s, %% updates Left: %s%%  Right: %s%%" % (round(np.mean(out), 3),round(np.std(out), 3),\
                round(np.mean(percent_left) * 100, 2), \
                round(np.mean(percent_right) * 100, 2))

        out, percent_left, percent_rel, percent_right = [], [], [], []
        timeref = time.time()
        
        if (epoch_count % state.test_all) == 0:
            ### evaluate by actually computing ranking metrics on some data
            if state.nvalid > 0:
                verl, verr, ver_rel = FilteredRankingScoreIdx(model.ranklfunc,\
                    model.rankrfunc, validlidx, validridx, validoidx, \
                    true_triples, rank_rel=model.rankrelfunc)
                state.valid = np.mean(verl + verr)# only tune on entity ranking
            else:
                state.valid = 'not applicable'
            
            if state.ntrain > 0:
                terl, terr, ter_rel = FilteredRankingScoreIdx(model.ranklfunc,\
                    model.rankrfunc, trainlidx, trainridx, trainoidx, \
                    true_triples, rank_rel=model.rankrelfunc)
                state.train = np.mean(terl + terr)# only tune on entity ranking
            else:
                state.train = 'not applicable'
            
            print >> sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (
                    round(state.valid, 1), round(state.train, 1))
            print >> sys.stderr, "\t\tMEAN RANK TRAIN: Left: %s Rel: %s Right: %s" % (round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1))
            print >> sys.stderr, "\t\tMEAN RANK VALID: Left: %s Rel: %s Right: %s" % (round(np.mean(verl), 1), round(np.mean(ver_rel), 1), round(np.mean(verr), 1))

            ### save model that performs best on dev data
            if state.bestvalid == -1 or state.valid < state.bestvalid:
                terl, terr, ter_rel = FilteredRankingScoreIdx(model.ranklfunc,\
                    model.rankrfunc, testlidx, testridx, testoidx,true_triples)
                state.bestvalid = state.valid
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
                print >> sys.stderr, "\tNEW BEST TEST: %s\n\t\tLeft: %s Rel: %s Right: %s" % (round(state.besttest, 1), round(np.mean(terl), 1), round(np.mean(ter_rel), 1), round(np.mean(terr), 1))
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(model.embeddings, f, -1)
            cPickle.dump(model.leftop, f, -1)
            cPickle.dump(model.rightop, f, -1)
            cPickle.dump(model.simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(ranking took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()
    print >> sys.stderr, "------------------------------------------------------"
    print >> sys.stderr, "------------------------------------------------------"
    return channel.COMPLETE

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def launch(experiment_type='FB15kexp', datapath='data/', dataset='FB15k', \
    Nent=16296, rhoE=1, margincostfunction='margincost', \
    rhoL=5, Nsyn_rel = 1345, Nsyn=14951, Nrel=1345, loadmodel=False, \
    loademb=False, op='Unstructured', simfn='Dot', ndim=50, marge=1., \
    lremb=0.1, lrparam=1., nbatches=100, totepochs=2000, test_all=1, \
    ntrain = 'all', nvalid = 'all', ntest = 'all', seed=123, \
    savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/FB15k_TransE/', loadmodelBi=False, \
    loadmodelTri=False, rel=False):

    # Argument of the experiment script
    state = DD()
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

    channel = Channel(state)
    FB15kexp(state, channel)

def launch_text(experiment_type='FB15kexp_text', datapath='data/', \
    dataset='FB15k', Nent=16296, rhoE=1, rhoL=5, Nsyn_rel = 1345, \
    Nsyn=14951, Nrel=1345, loadmodel=False, margincostfunction='margincost', \
    loademb=False, op='Unstructured', simfn='Dot', ndim=50, marge=1., \
    lremb=0.1, lrparam=1., nbatches=100, totepochs=2000, test_all=1, \
    ntrain = 'all', nvalid = 'all', ntest = 'all', textsim = 'L2', \
    vocab_size = 100000, word_dim = 300, word_file = None, vocab = None, \
    gamma = 0.01, seed=123, savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/FB15k_TransE/', loadmodelBi=False, \
    loadmodelTri=False, rel=False, numTextTrain=10000):

    # Argument of the experiment script
    state = DD()
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
    state.ntrain = ntrain # num exs from train to compute a rank for, 
                          # 'all' or some num
    state.nvalid = nvalid # num exs from valid to compute a rank for
    state.ntest = ntest   # num exs from test to compute a rank for
    state.seed = seed

    if not os.path.isdir(state.savepath):
        os.mkdir(state.savepath)

    channel = Channel(state)
    FB15kexp_text(state, channel)


if __name__ == '__main__':
    launch()
