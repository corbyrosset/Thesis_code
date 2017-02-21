#! /usr/bin/python
import sys
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/models/")
from KBC_models import *
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/evaluation/")
from evaluate_KBC import *
from Utils import * 

# Experiment function --------------------------------------------------------
def FB15kexp(state, channel):
    '''
        Main training loop

        out = list of costs per minibatch for this epoch
        outb = list of percentages of ??? that were updated per minibatch in this epoch??
        restrain & state.train = ranks of all epoch train triples, and the mean thereof
        resvalid & state.valid = ranks of all epoch dev triples, and the mean thereof

    '''
    out = []
    outb = []
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
    validridx, validoidx, testlidx, testridx, testoidx, true_triples \
    = load_FB15k_data(state)

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
    print 'left and right entity ranking functions will rank a triple against ' + str(state.Nsyn) + ' competitors'

    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        
        # Negatives, TODO, these should be filtered as well?
        trainln = create_random_mat(trainl.shape, np.arange(state.Nsyn))
        trainrn = create_random_mat(trainr.shape, np.arange(state.Nsyn))
        if state.rel == True: ### create negative relationship instances
            trainon = create_random_mat(traino.shape, np.arange(state.Nsyn_rel))

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
                outtmp = model.trainfunc(state.lremb, state.lrparam,
                        tmpl, tmpr, tmpo, tmpnl, tmpnr, tmpno)
                out += [outtmp[0] / float(batchsize)]
                outb += [outtmp[1]]
            else:
                outtmp = model.trainfunc(state.lremb, state.lrparam,
                        tmpl, tmpr, tmpo, tmpnl, tmpnr)
                out += [outtmp[0] / float(batchsize)]
                outb += [outtmp[1]]


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
        print >> sys.stderr, "\tCOST >> %s +/- %s, %% updates: %s%%" % (
                round(np.mean(out), 4), round(np.std(out), 4),
                round(np.mean(outb) * 100, 3))
        out = []
        outb = []
        timeref = time.time()
        
        if (epoch_count % state.test_all) == 0:
            ### evaluate by actually computing ranking metrics on some data
            if state.nvalid > 0:
                resvalid = FilteredRankingScoreIdx(model.ranklfunc, \
                    model.rankrfunc, validlidx, validridx, validoidx, \
                    true_triples, rank_rel=model.rankrelfunc)
                state.valid = np.mean(resvalid[0] + resvalid[1])
            else:
                state.valid = 'not applicable'
            if state.ntrain > 0:
                restrain = FilteredRankingScoreIdx(model.ranklfunc, \
                    model.rankrfunc, trainlidx, trainridx, trainoidx, \
                    true_triples, rank_rel=model.rankrelfunc)
                state.train = np.mean(restrain[0] + restrain[1])
            else:
                state.train = 'not applicable'
            
            print >> sys.stderr, "\tMEAN RANK >> valid: %s, train: %s" % (
                    state.valid, state.train)

            ### save model that performs best on dev data
            if state.bestvalid == -1 or state.valid < state.bestvalid:
                restest = FilteredRankingScoreIdx(model.ranklfunc, model.rankrfunc, testlidx, testridx, testoidx, true_triples)
                state.bestvalid = state.valid
                state.besttrain = state.train
                state.besttest = np.mean(restest[0] + restest[1])
                state.bestepoch = epoch_count
                # Save model best valid model
                f = open(state.savepath + '/best_valid_model.pkl', 'w')
                cPickle.dump(model.embeddings, f, -1)
                cPickle.dump(model.leftop, f, -1)
                cPickle.dump(model.rightop, f, -1)
                cPickle.dump(model.simfn, f, -1)
                f.close()
                print >> sys.stderr, "\t\tNEW BEST VALID >> test: %s" % (
                        state.besttest)
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


def launch(datapath='data/', dataset='FB15k', Nent=16296, rhoE=1, \
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

    channel = Channel(state)

    FB15kexp(state, channel)

if __name__ == '__main__':
    launch()
