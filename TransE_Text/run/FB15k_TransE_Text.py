#! /usr/bin/python
import sys
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/utils")
from FB15k_exp import *
sys.path.append("/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/evaluation")
from evaluate_KBC import RankingEval

###############################################################################
###############################################################################
datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/'
simfn = 'L2'
margincostfunction = 'margincost' ### from top of Operations
ndim = 300 # dimension of both relationship and entity embeddings
	       # {10, 50, 100, 150}
marge = 0.5     # {0.5, 1.0}
lremb = 0.01   # {0.01, 0.001}
lrparam = 0.01 # {0.01, 0.001}
nbatches = 100  # number of batches per epoch
totepochs = 50  # number of epochs should be 500
test_all = 10   # number of epochs between ranking on validation sets again
Nsyn = 14951    # number of entities against which to rank a given test
			    ### TODO: doesn't work if < 14951
Nsyn_rel = 1345 # only matters if rel = True, number of relations to rank for 
				# a triple with missing relationship
rel = True   # whether to also rank relations

### although these should be higher numbers (preferably 'all'), it would
### take too long, and with these numbers we can at least compare to 
### previous runs...
ntrain = 1000 # 'all' # number of examples to actually compute ranks for
		      # if you set to 'all', it will take a very long time
nvalid = 1000 # 'all'
ntest = 1000 # 'all'
neval = 'all' # 'all'### only for final testing, not training

###############################################################################
### parameters specific for textual triples. 
textsim = 'L2' # how to compare a textual relation to KB relation
vocab_size = 354936 # size of vocabulary
word_dim = 300 # dimension of each word embedding
word_file = '/Users/corbinrosset/Dropbox/Paragrams/paragrams-XXL-SL999.txt'
	# path to file containing word embeddings
vocab = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/process_clueweb/dictionary.txt'
gamma = 0.1 #{0.01, 0.1, 1} weight to use for cost of textual triple
# assert ndim == word_dim
numTextTrain = 100000 # num textual triples to use in each epoch of training

###############################################################################
###############################################################################
# for the word-averaging model of sentence embeddings, 
assert word_dim == ndim ### else can't compare sentence and entity embeddings

savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/FB15k_TransE_Text/'

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
launch_text(experiment_type = 'FB15k_text', op='TransE_text', simfn= simfn, \
	ndim= ndim, marge= marge, margincostfunction=margincostfunction, \
	lremb= lremb, lrparam= lrparam, nbatches= nbatches, totepochs= totepochs,\
	test_all= test_all, Nsyn=Nsyn, Nsyn_rel=Nsyn_rel, \
	savepath= savepath + str(identifier), numTextTrain = numTextTrain, \
	ntrain=ntrain, nvalid=nvalid, ntest=ntest, dataset='FB15k', rel=rel, \
	textsim = textsim, vocab_size = vocab_size, vocab = vocab, \
	word_dim=word_dim, word_file=word_file, gamma = gamma, datapath = datapath)

### evaluate on test data, always set neval to 'all' to rank all test triples
### this will take a couple hours to run...

RankingEval(neval=neval, loadmodel= savepath + str(identifier) \
	+ '/best_valid_model.pkl', Nsyn=Nsyn, rel=rel, Nsyn_rel=Nsyn_rel)
RankingEvalFil(neval=neval, loadmodel= savepath + str(identifier) + \
	'/best_valid_model.pkl', Nsyn=Nsyn, rel=rel, Nsyn_rel=Nsyn_rel)

###############################################################################
###############################################################################