#! /usr/bin/python
import sys
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/utils")
from FB15k_exp import *
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/evaluation")
from evaluate_KBC import RankingEval

###############################################################################
###############################################################################

# launch(op='TransE', simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01,
#    nbatches=100, totepochs=500, test_all=10, neval=1000, savepath='FB15k_TransE', datapath='../data/', dataset='FB15k')
simfn = 'L2'
margincostfunction = 'squared_margin_cost' ### from top of Operations
ndim = 150 # dimension of both relationship and entity embeddings
	       # {10, 50, 100, 150}
marge = 0.5     # {0.5, 1.0} 
lremb = 0.01   # {0.01, 0.001}
lrparam = 0.01 # {0.01, 0.001}
nbatches = 100  # number of batches per epoch
totepochs = 500 # number of epochs
test_all = 10   # number of epochs between ranking on validation sets again
Nsyn = 14951    # number of entities against which to rank a given test
			    ### TODO: doesn't work if < 14951
Nsyn_rel = 1345 # only matters if rel = True, number of relations to rank for 
				# a triple with missing relationship
rel = False   # whether to also rank relations
reg = 0.01

### although these should be higher numbers (preferably 'all'), it would
### take too long, and with these numbers we can at least compare to 
### previous runs...
ntrain = 1000 # 'all' # number of examples to actually compute ranks for
		      # if you set to 'all', it will take a very long time
nvalid = 1000 # 'all'
ntest = 1000 # 'all'
neval = 'all' # 'all'### only for final testing, not training

savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/FB15k_TransE/'
datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/'

if rel == True:
	identifier = 'TransE_' + str(simfn) + '_ndim_' + str(ndim) \
		+ '_marg_' + str(marge) + '_lrate_' + str(lremb) + '_cost_' + str(margincostfunction) + '_REL'
else:
	identifier = 'TransE_' + str(simfn) + '_ndim_' + str(ndim) \
		+ '_marg_' + str(marge) + '_lrate_' + str(lremb) + '_cost_' + str(margincostfunction)

###############################################################################
###############################################################################

print 'identifier: ' + str(identifier)
print 'models saved to path: ' + str(savepath)
launch(experiment_type = 'FB15kexp', op='TransE', simfn= simfn, ndim= ndim, \
	marge= marge, margincostfunction=margincostfunction, \
	lremb= lremb, lrparam= lrparam, nbatches= nbatches, totepochs= totepochs,\
	test_all= test_all, Nsyn=Nsyn, Nsyn_rel=Nsyn_rel, \
	savepath= savepath + str(identifier), \
	ntrain=ntrain, nvalid=nvalid, ntest=ntest, dataset='FB15k', rel=rel, \
	datapath=datapath)

### evaluate on test data, always set neval to 'all' to rank all test triples
### this will take a couple hours to run...

RankingEval(datapath=datapath, neval=neval, loadmodel= savepath + \
	str(identifier) + '/best_valid_model.pkl', Nsyn=Nsyn, rel=rel, \
	Nsyn_rel=Nsyn_rel)
RankingEvalFil(datapath=datapath, neval=neval, loadmodel= savepath + \
	str(identifier) + '/best_valid_model.pkl', Nsyn=Nsyn, rel=rel, \
	Nsyn_rel=Nsyn_rel)
###############################################################################
###############################################################################
### notes:
'''
calls FB15k_exp.launch()
	- which then calls FB15k_exp.FB15kexp()

	- FB15kexp():
		creates left and right ops, for TransE this is
		LayerTrans and Unstructured = identiy
		
		also creates random/loads embedding matrices 
		embeddings = Embeddings(np.random, state.Nent, state.ndim) ENTITIES
		relationVec = Embeddings(np.random, state.Nrel, state.ndim, 'relvec')
        embeddings = [embeddings, relationVec, relationVec] meaning that left
        and right relations are equal, there is only one relaitno paramter

        cretes/evals similiarty function between left and right; from model.py

        creates/compiles training and ranking functions:
        trainfunc = TrainFn1Member(simfn, embeddings, leftop, rightop,
                marge=state.marge, rel=False)
        ranklfunc = RankLeftFnIdx(simfn, embeddings, leftop, rightop,
                subtensorspec=state.Nsyn)
        rankrfunc = RankRightFnIdx(simfn, embeddings, leftop, rightop,
                subtensorspec=state.Nsyn)

	Controls training iterations:

	outtmp = trainfunc(state.lremb, state.lrparam, tmpl, tmpr, tmpo, tmpnl, tmpnr) ### what is outtmp:
	    :output mean(cost): average cost.
    	:output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.

	evaluation on dev data: FilteredRankingScoreIdx()

Model.py:
	- line 1070: no adam optimizer or anything just SGD. 
FB15k_evaluation:
	- RankingEval():

'''

