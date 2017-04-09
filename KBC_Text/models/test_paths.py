from Operations import compose_TransE, compose_BilinearDiag, compose_TransE_noinit, compose_BilinearDiag_noinit
import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T


# The order of arguments to scan's lambda function is as follows:

# First, the current items from the variables that we are iterating over. If 
	#we are iterating over a matrix, the current row is passed to the function.
# Next, anything that was output from the function at the previous time step. 
	# This is what we use to build recursive and recurrent representations. At 
	# the very first time step, the values will be those from outputs_info 
	# instead.
# Finally, anything we specified in non_sequences.

def test_path():
	is_horn_path = True
	dimension = 3
	W_values = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=theano.config.floatX).T # each of the 4 relation is 3-dimensions
	rels = theano.shared(value=W_values, name='rels')
	initial_vecs = T.matrix("initial")
	idxs_per_path = T.imatrix("indexes")
	
	idxs_per_path_python = np.asarray([[0, 0], [0, 1], [1, 2]], dtype=np.int32)
	initial_vecs_python = np.ones((idxs_per_path_python.shape[0], dimension), dtype=theano.config.floatX)
	
	# Symbolic description of the result
	if not is_horn_path:
		### we want to ignore previous steps of the outputs.
		### idxs_per_path is a matrix of N paths, each length L
		### we are mapping each row of idxs_per_path to a D-dim vector 
		### embedding of the path. Initial_vecs is N by D, the initial 
		### embedding of the path (the start entity vector in non-horn paths). 
		### rels is the D by numRels embedding matrix of relationships, fixed 

		### because these are not horn paths, initial vectors must be supplied 
		### by user during training
		res, updates = theano.map(
		                fn= compose_BilinearDiag, #compose_TransE,
		                sequences=[idxs_per_path, initial_vecs],
		                non_sequences=[rels])

		### apparently pathvec is of shape [N by D] for N paths
		output = theano.function(inputs=[idxs_per_path, initial_vecs], outputs=[res])
		print output(idxs_per_path_python, initial_vecs_python)

	else:
		# paths start at first relation, compose relations until finished.
		# final result should be close to the intended cycle-completing rel

		### be mindful what to set initial vectors to - 0s for transE, 
		### 1s for bilinear
		initial_vecs = np.ones((idxs_per_path_python.shape[0], dimension)
			, dtype=theano.config.floatX)
		initial_vecs_fixed = theano.shared(value=initial_vecs, name='init')

		res, updates = theano.map(
		                fn= compose_BilinearDiag, #compose_TransE,
		                sequences=[idxs_per_path, initial_vecs_fixed],
		                non_sequences=[rels])
		output = theano.function(inputs=[idxs_per_path], outputs=[res])
		print output(idxs_per_path_python)

	

test_path()