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
margincostfunction = 'margincost_pos_high' ### from top of Operations
reverseRanking = True # rank from best -> worst <=> high to low score

ndim = 50 # dimension of both relationship and entity embeddings
	       # {10, 50, 100, 150, 200}
marge = 0.2     # {0.5, 1.0} 
lremb = 0.01    # {0.01, 0.001}
lrparam = 0.01  # {0.01, 0.001}
nbatches = 100  # number of batches per epoch
totepochs = 500 # number of epochs
test_all = 10   # number of epochs between ranking on validation sets again
Nsyn = 14951    # number of entities against which to rank a given test
			    ### TODO: doesn't work if < 14951
Nsyn_rel = 1345 # only matters if rel = True, number of relations to rank for 
				# a triple with missing relationship
rel = False     # This must be false for Model E because we can't sample
				# 'negative' two part relations...
reg = 0.1       #{0.01, 0.1} if None, no regularization (= 0.0)

### although these should be higher numbers (preferably 'all'), it would
### take too long, and with these numbers we can at least compare to 
### previous runs...
ntrain = 1000 # 'all' # number of examples to actually compute ranks for
		      # after every test_all'th epoch in training
nvalid = 1000 # 'all'
ntest = 1000  # 'all'
neval = 'all' # 'all'### only for final testing, not training

savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/FB15k_BilinearDiagExtended/'
datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/'

identifier = 'BilinearDiagExtended_' + str(simfn) + '_ndim_' + str(ndim) \
		+ '_marg_' + str(marge) + '_lrate_' + str(lremb) + '_cost_' + str(margincostfunction) + '_reg_' + str(reg)
if rel == True:
	identifier += '_REL'
	

###############################################################################
###############################################################################

print 'identifier: ' + str(identifier)
print 'models saved to path: ' + str(savepath)
launch(experiment_type = 'FB15kexp', op='BilinearDiagExtended', simfn= simfn, \
	ndim= ndim, marge= marge, margincostfunction=margincostfunction, \
	lremb= lremb, lrparam= lrparam, nbatches= nbatches, totepochs= totepochs,\
	test_all= test_all, Nsyn=Nsyn, Nsyn_rel=Nsyn_rel, \
	savepath= savepath + str(identifier), reg=reg, \
	ntrain=ntrain, nvalid=nvalid, ntest=ntest, dataset='FB15k', rel=rel, \
	datapath=datapath)

### evaluate on test data, always set neval to 'all' to rank all test triples
### this will take a couple hours to run...

RankingEval(datapath, reverseRanking=True, neval=neval, loadmodel= savepath + \
	str(identifier) + '/best_valid_model.pkl', Nsyn=Nsyn, rel=rel, \
	Nsyn_rel=Nsyn_rel)
RankingEvalFil(datapath, reverseRanking=True, neval=neval, loadmodel=savepath+\
	str(identifier) + '/best_valid_model.pkl', Nsyn=Nsyn, rel=rel, \
	Nsyn_rel=Nsyn_rel)
###############################################################################
###############################################################################

