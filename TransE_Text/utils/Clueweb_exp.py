#! /usr/bin/python
import sys
sys.path.append("../models/")
from KBC_models import *
# from text_encoders import *
sys.path.append("../evaluation")
from evaluate_Clueweb import FilteredRankingScoreIdx
from Utils import * 


# Experiment function --------------------------------------------------------
def Clueweb_exp(state, channel):

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

    out = []
    outb = []
    state.bestvalid = -1

    batchsize = trainl.shape[1] / state.nbatches

    print >> sys.stderr, "BEGIN TRAINING"
    timeref = time.time()
    for epoch_count in xrange(1, state.totepochs + 1):
        # Shuffling
        order = np.random.permutation(model.trainl.shape[1])
        trainl = trainl[:, order]
        trainr = trainr[:, order]
        traino = traino[:, order]
        
        # Negatives
        trainln = create_random_mat(model.trainl.shape, np.arange(state.Nsyn))
        trainrn = create_random_mat(model.trainr.shape, np.arange(state.Nsyn))

        for i in range(state.nbatches):
            tmpl = trainl[:, i * batchsize:(i + 1) * batchsize]
            tmpr = trainr[:, i * batchsize:(i + 1) * batchsize]
            tmpo = traino[:, i * batchsize:(i + 1) * batchsize]
            tmpnl = trainln[:, i * batchsize:(i + 1) * batchsize]
            tmpnr = trainrn[:, i * batchsize:(i + 1) * batchsize]
            
            # training iteration
            outtmp = model.trainfunc(state.lremb, state.lrparam,
                    tmpl, tmpr, tmpo, tmpnl, tmpnr)
            out += [outtmp[0] / float(batchsize)]
            outb += [outtmp[1]]

            # embeddings normalization
            model.embeddings.normalize()

        if (epoch_count % state.test_all) == 0:
            # model evaluation
            print >> sys.stderr, "-- EPOCH %s (%s seconds per epoch):" % (
                    epoch_count, round(time.time() - timeref, 3) / \
                    float(state.test_all))
            timeref = time.time()
            print >> sys.stderr, "COST >> %s +/- %s, %% updates: %s%%" % (
                    round(np.mean(out), 4), round(np.std(out), 4),
                    round(np.mean(outb) * 100, 3))
            out = []
            outb = []

            ### evaluate train and dev data
            resvalid = FilteredRankingScoreIdx(model.ranklfunc, model.rankrfunc, validlidx, validridx, validoidx, true_triples)
            state.valid = np.mean(resvalid[0] + resvalid[1])
            restrain = FilteredRankingScoreIdx(model.ranklfunc, model.rankrfunc , trainlidx, trainridx, trainoidx, true_triples)
            state.train = np.mean(restrain[0] + restrain[1])
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
                cPickle.dump(embeddings, f, -1)
                cPickle.dump(leftop, f, -1)
                cPickle.dump(rightop, f, -1)
                cPickle.dump(simfn, f, -1)
                f.close()
                print >> sys.stderr, "\t\t##### NEW BEST VALID >> test: %s" % (
                        state.besttest)
            # Save current model
            f = open(state.savepath + '/current_model.pkl', 'w')
            cPickle.dump(embeddings, f, -1)
            cPickle.dump(leftop, f, -1)
            cPickle.dump(rightop, f, -1)
            cPickle.dump(simfn, f, -1)
            f.close()
            state.nbepochs = epoch_count
            print >> sys.stderr, "\t(the evaluation took %s seconds)" % (
                round(time.time() - timeref, 3))
            timeref = time.time()
            channel.save()
    return channel.COMPLETE


def launch(datapath='data/', dataset='FB15k', Nent=16296, rhoE=1, \
    rhoL=5,
    Nsyn=14951, Nrel=1345, loadmodel=False, loademb=False, \
    op='Unstructured', simfn='Dot', ndim=50, nhid=50, marge=1., \
    lremb=0.1, lrparam=1., nbatches=100, totepochs=2000, test_all=1, \
    neval=50, seed=123, savepath='.', loadmodelBi=False, \
    loadmodelTri=False):

    # Argument of the experiment script
    state = DD()
    state.datapath = datapath
    state.dataset = dataset
    state.Nent = Nent
    state.Nsyn = Nsyn
    state.Nrel = Nrel
    state.loadmodel = loadmodel
    state.loadmodelBi = loadmodelBi
    state.loadmodelTri = loadmodelTri
    state.loademb = loademb ### load previously trained embeddings?
    state.op = op
    state.simfn = simfn
    state.ndim = ndim
    state.nhid = nhid
    state.marge = marge
    state.rhoE = rhoE
    state.rhoL = rhoL
    state.lremb = lremb
    state.lrparam = lrparam
    state.nbatches = nbatches
    state.totepochs = totepochs
    state.test_all = test_all
    state.neval = neval ### number of examples for train, dev, test?
    state.seed = seed
    state.savepath = savepath

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
