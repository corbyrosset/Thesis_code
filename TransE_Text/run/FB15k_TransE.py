#! /usr/bin/python
import sys
sys.path.append("../utils")
from FB15k_exp import *
sys.path.append("../evaluation")
from evaluate_KBC import RankingEval


# totepochs = 500 default
# try L1 similarity

# launch(op='TransE', simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01,
#    nbatches=100, totepochs=500, test_all=10, neval=1000, savepath='FB15k_TransE', datapath='../data/', dataset='FB15k')
simfn = 'L2'
ndim = 150
nhid = 150
marge = 0.5
lremb = 0.01
lrparam = 0.01
nbatches = 100
totepochs = 500
test_all = 10
neval = 1000
savepath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/FB15k_TransE/'

identifier = 'TransE_' + str(simfn) + '_ndim_' + str(ndim) + '_nhid_' + str(nhid) + '_marg_' + str(marge)
print 'identifier: ' + str(identifier)
print 'models saved to path: ' + str(savepath)
launch(op='TransE', simfn= simfn, ndim= ndim, nhid= nhid, marge= marge, \
	lremb= lremb, lrparam= lrparam, nbatches= nbatches, totepochs= totepochs,\
	test_all= test_all, neval= neval, \
	savepath= savepath + str(identifier), \
	datapath='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/data/',
	dataset='FB15k')

### evaluate on test data:
RankingEval(loadmodel='/Users/corbinrosset/Dropbox/Arora/QA-code/src/TransE_Text/outputs/' + str(identifier) + 'best_valid_model.pkl')


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

