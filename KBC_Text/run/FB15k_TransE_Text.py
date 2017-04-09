#! /usr/bin/python
import sys
import os
from KBC_Text.utils.FB15k_exp import launch_text
from KBC_Text.utils.Utils import initialize_logging
from KBC_Text.utils.send_email import send_notification
from KBC_Text.evaluation.evaluate_KBC import RankingEval, RankingEvalFil

###############################################################################
###############################################################################
datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/data/'
savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/KBC_Text/outputs/FB15k_TransE_Text/'

simfn = 'L2'
margincostfunction = 'margincost' ### from top of Operations
ndim = 100 # dimension of both relationship and entity embeddings
	      # {10, 50, 100, 150}
marge = 0.5     # {0.5, 1.0}
lremb = 0.01    # {0.01, 0.001}
lrparam = 0.01  # {0.01, 0.001}
nbatches = 100  # number of batches per epoch
totepochs = 300  # number of epochs should be 500
test_all = 10    # number of epochs between ranking on validation sets again
Nsyn = 14951    # number of entities against which to rank a given test
			    ### TODO: doesn't work if < 14951
Nsyn_rel = 1345 # only matters if rel = True, number of relations to rank for 
				# a triple with missing relationship
rel = False   # whether to also rank relations
reg = 0.01       #{0.01, 0.1} if None, no regularization (= 0.0)


### although these should be higher numbers (preferably 'all'), it would
### take too long, and with these numbers we can at least compare to 
### previous runs...
ntrain = 1000 # 'all' # number of examples to actually compute ranks for
		      # if you set to 'all', it will take a very long time
nvalid = 1000 # 'all'
ntest = 1000 # 'all'
neval = 'all' # 'all'### only for final testing, not training
experiment_type = 'FB15k_text'

###############################################################################
### parameters specific for textual triples.
textual_role = 'TextAsRegularizer' # {TextAsRegularizer, TextAsRelation, TextAsRelAndReg}
marg_text = 2.0
textsim = 'L2' # how to compare a textual relation to KB relation
vocab_size = 354936 # size of vocabulary
word_dim = 100 # dimension of each word embedding
# word_file = '/Users/corbinrosset/Dropbox/Paragrams/paragrams-XXL-SL999.txt'
word_file = '/Users/corbinrosset/Dropbox/GloVe/glove.6B/glove.6B.100d.txt'
	# path to file containing word embeddings
vocab = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/process_clueweb/dictionary.txt'
gamma = 1.0 #{0.01, 0.1, 1} weight to use for cost of textual triple
# assert ndim == word_dim
numTextTrain = 1000000 # num textual triples to use in each epoch of training
					   # maximum is 10413174
# for the word-averaging model of sentence embeddings, 
assert word_dim == ndim ### else can't compare sentence and entity embeddings

###############################################################################
###############################################################################
identifier = 'TransE_Text_' + str(simfn) + '_ndim_' + str(ndim) \
		+ '_marg_' + str(marge) + '_textmarg_' + str(marg_text) + '_lrate_' + \
		str(lremb) + '_cost_' + str(margincostfunction) + '_role_' + \
		str(textual_role)
if rel == True:
	identifier += '_REL'

# I hate to do this here, but it is needed for .log to be in right place
if not os.path.isdir(savepath + identifier + '/'):
	os.makedirs(savepath + identifier + '/')
logger, logFile = initialize_logging(savepath + identifier + '/', identifier)

###############################################################################
###############################################################################

logger.info('identifier: ' + str(identifier))
logger.info('models saved to path: ' + str(savepath))
launch_text(identifier, experiment_type, logger, op='TransE_text', \
	simfn= simfn, reg=reg, \
	ndim= ndim, marge= marge, margincostfunction=margincostfunction, \
	lremb= lremb, lrparam= lrparam, nbatches= nbatches, totepochs= totepochs,\
	test_all= test_all, Nsyn=Nsyn, Nsyn_rel=Nsyn_rel, \
	savepath= savepath + str(identifier), numTextTrain = numTextTrain, \
	ntrain=ntrain, nvalid=nvalid, ntest=ntest, dataset='FB15k', rel=rel, \
	textsim = textsim, vocab_size = vocab_size, marg_text=marg_text, \
	vocab = vocab, word_dim=word_dim, word_file=word_file, gamma = gamma,\
	datapath = datapath, textual_role=textual_role)

### evaluate on test data, always set neval to 'all' to rank all test triples
### this will take a couple hours to run...

RankingEval(datapath, logger, reverseRanking=False, neval=neval, \
	loadmodel= savepath + str(identifier) + '/best_valid_model.pkl', \
	Nsyn=Nsyn, rel=rel, Nsyn_rel=Nsyn_rel)
RankingEvalFil(datapath, logger, reverseRanking=False, neval=neval, \
	loadmodel= savepath + str(identifier) + '/best_valid_model.pkl', \
	Nsyn=Nsyn, rel=rel, Nsyn_rel=Nsyn_rel)
# send_notification(identifier, logFile)

###############################################################################
###############################################################################