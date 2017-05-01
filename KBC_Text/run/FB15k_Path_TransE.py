#! /usr/bin/python
import sys
import os
from KBC_Text.utils.FB15k_exp import launch_path
from KBC_Text.utils.Utils import initialize_logging
from KBC_Text.utils.send_email import send_notification
from KBC_Text.evaluation.evaluate_KBC import RankingEval, RankingEvalFil

###############################################################################
###############################################################################

simfn =  'L1'
margincostfunction = 'margincost' 
compop = 'compose_TransE'
ndim = 100 # dimension of both relationship and entity embeddings
	       # {10, 50, 100, 150, 200}
marge = 1.5     # {0.5, 1.0} 
lremb = 0.01    # {0.01, 0.001}
lrparam = 0.01  # {0.01, 0.001}
nbatches = 1000  # number of batches per epoch
totepochs = 500 # number of epochs
test_all = 3   # number of epochs between ranking on validation sets again
Nsyn = 14951    # number of entities against which to rank a given test
			    ### TODO: doesn't work if < 14951
Nsyn_rel = 1345 # only matters if rel = True, number of relations to rank for 
				# a triple with missing relationship
rel = False      # whether to also rank relations
reg = 0.002       #{0.01, 0.1} if None, no regularization (= 0.0)

### although these should be higher numbers (preferably 'all'), it would
### take too long, and with these numbers we can at least compare to 
### previous runs...
ntrain = 1000 # 'all' # number of examples to actually compute ranks for
		      # if you set to 'all', it will take a very long time
nvalid = 1000 # 'all'
ntest = 1000 # 'all'
neval = 'all' # 'all'### only for final testing, not training
experiment_type = 'FB15k_path_exp'
savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/outputs/FB15k_Path_TransE/'
datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/data/'

### these are prefixes to the train, dev, test splits of the .path files
graph_files = ['length_3_numPaths_50000000', 'length_4_numPaths_50000000'] #['length_2_numPaths_50000000', 'length_3_numPaths_50000000']

### for these two flags, see graph/Graph.py
useHornPaths = False 
needIntermediateNodesOnPaths = False
# loademb = None ### initialize with pre-trained vectors??
loademb = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/outputs/FB15k_TransE/' + 'BEST_TransE_L1_ndim_100_marg_1.5_lrate_0.01_cost_margincost_reg_0.01_REL' + '/best_valid_model.pkl'



###############################################################################
# DONT TOUCH BETWEEN HERE
###############################################################################

identifier = 'Path_TransE_' + str(simfn) + '_ndim_' + str(ndim) + \
			'_marg_' + str(marge) + '_lrate_' + str(lremb) + '_cost_' + \
			str(margincostfunction) + '_reg_' + str(reg) + '_horn_' + \
			str(useHornPaths) + '_useIntermedNodes_' + \
			str(needIntermediateNodesOnPaths) + '_compop_' + str(compop) + \
			'_loadEmb_' + str(True if loademb else False) + \
			'_1kbatches_train1M_length3and4only'

if rel == True:
	identifier += '_REL'

# I hate to do this here, but it is needed for .log to be in right place
if not os.path.isdir(savepath + identifier + '/'):
	os.makedirs(savepath + identifier + '/')
logger, logFile = initialize_logging(savepath + identifier + '/', identifier)

if 'TransE' in compop:
	reverseRanking = False
else:
	reverseRanking = True
	
###############################################################################
# AND HERE
###############################################################################

### comment out this part if you just want to evaluate an already-trained model
logger.info('identifier: ' + str(identifier))
logger.info('models saved to path: ' + str(savepath))
launch_path(identifier, experiment_type, logger, \
	ndim= ndim, compop=compop, simfn= simfn, \
	marge= marge, margincostfunction=margincostfunction, \
	lremb= lremb, lrparam= lrparam, nbatches= nbatches, totepochs= totepochs,\
	test_all= test_all, Nsyn=Nsyn, Nsyn_rel=Nsyn_rel, \
	savepath= savepath + str(identifier), reg=reg, \
	ntrain=ntrain, nvalid=nvalid, ntest=ntest, dataset='FB15k', rel=rel, \
	datapath=datapath, graph_files=graph_files, useHornPaths=useHornPaths, \
	needIntermediateNodesOnPaths=needIntermediateNodesOnPaths, \
	loademb = loademb)

### evaluate on test data, always set neval to 'all' to rank all test triples
### this will take a couple hours to run...

# send_notification(identifier, logFile)
RankingEval(datapath, logger, reverseRanking=reverseRanking, neval=neval, \
	loadmodel= savepath + str(identifier) + '/best_valid_model.pkl', \
	Nsyn=Nsyn, rel=rel, Nsyn_rel=Nsyn_rel)
RankingEvalFil(datapath, logger, reverseRanking=reverseRanking, neval=neval,\
	loadmodel= savepath + str(identifier) + '/best_valid_model.pkl', \
	Nsyn=Nsyn, rel=rel, Nsyn_rel=Nsyn_rel)
# send_notification(identifier, logFile)

###############################################################################
###############################################################################

