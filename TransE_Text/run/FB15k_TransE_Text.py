#! /usr/bin/python
import sys
sys.path.append("..")
from Clueweb_exp import *


# totepochs = 500 default
# try L1 similarity

launch(op='TransE_text', simfn='L2', ndim=50, nhid=50, marge=0.5, lremb=0.01, lrparam=0.01,
    nbatches=100, totepochs=500, test_all=10, neval=1000, savepath='FB15k_TransE_Text', datapath='../cluweb_data/', dataset='FB15k_Clueweb')


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

