import pickle, sys
sys.path.append('../src')
import data_io # eval
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD

## run
### the following is the word embeddings maps to use, download them pretrained
wordfiles = [#'../data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
    # '/Users/corbinrosset/Dropbox/GloVe/glove.840B.300d.txt'
    '/Users/corbinrosset/Dropbox/Paragrams/paragrams-XXL-SL999.txt'

    ]
weightfile = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/WordEmdgs/SIF/auxiliary_data/enwiki_vocab_min200.txt'
weightparas = [-1, 1e-3] #[-1,1e-1,1e-2,1e-3,1e-4] # (0.01 works best)
weightparas = [-1,1e-1,1e-2,1e-3,1e-4] # (0.01 works best)

rmpcs = [0, 1, 2, 3, 4] # remove the first k principle components, (4 is best)
scoring_function = None
save_result = False 
data_dir = '../data/original_and_standardized_data/'
paraphrases_file = '../data/combined_test_data_normed' ### data to evaluate
farr = ["MSRpar2012_normed",            # good
        "OnWN2012_normed",             # good
        "SMTeuro2012_normed",          # good 
        "SMTnews2012_normed",          # good
        "FNWN2013_normed",             # good
        "OnWN2013_normed",             # good
        "headlines2013_normed",        # good
        "OnWN2014_normed",             # good
        "deft-forum2014_normed",       # good
        "deft-news2014_normed",        # good
        "headlines2014_normed",        # good
        "images2014_normed",           # good
        "tweet-news2014_normed", # 14  # good
        "answers-forum2015_normed",    # good
        "answers-student2015_normed",  # good
        "belief2015_normed",           # good
        "headlines2015_normed",        # good
        "images2015_normed",    # 19   # good
        "sicktest_normed",             # good
        "annotated-ppdb-dev_normed",   # good
        "annotated-ppdb-test_normed"]  # good


###############################################################################
###                         Helper Classes                                  ###
###############################################################################

class params(object):
    
    def __init__(self):
        self.LW = 1e-5
        self.LC = 1e-5
        self.eta = 0.05

    def __str__(self):
        t = "LW", self.LW, ", LC", self.LC, ", eta", self.eta
        t = map(str, t)
        return ' '.join(t)

###############################################################################
###                         Helper Methods                                  ###
###############################################################################

def evaluate_similarity(We,words,f, weight4ind, scoring_function, params):
    golds = []
    seq1 = []
    seq2 = []
    with open(f,'r') as fl:
        for i in fl:
            i = i.split("\t")
            p1 = i[0]; p2 = i[1]; score = float(i[2])
            X1, X2 = data_io.getSeqs(p1,p2,words)
            seq1.append(X1)
            seq2.append(X2)
            golds.append(score)
    x1,m1 = data_io.prepare_data(seq1)
    x2,m2 = data_io.prepare_data(seq2)
    m1 = data_io.seq2weight(x1, m1, weight4ind)
    m2 = data_io.seq2weight(x2, m2, weight4ind)
    scores = scoring_function(We,x1,x2,m1,m2, params)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]


###############################################################################
###                      SIF Averaging Methods                              ###
###############################################################################

def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in xrange(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb

def compute_pc(X,npc=1):
    """
    Compute the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def weighted_average_sim_rmpc(We,x1,x2,w1,w2, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x1: x1[i, :] are the indices of the words in the first sentence in pair i
    :param x2: x2[i, :] are the indices of the words in the second sentence in pair i
    :param w1: w1[i, :] are the weights for the words in the first sentence in pair i
    :param w2: w2[i, :] are the weights for the words in the first sentence in pair i
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: scores, scores[i] is the matching score of the pair i
    """
    emb1 = get_weighted_average(We, x1, w1)
    emb2 = get_weighted_average(We, x2, w2)
    if  params.rmpc > 0:
        emb1 = remove_pc(emb1, params.rmpc)
        emb2 = remove_pc(emb2, params.rmpc)

    inn = (emb1 * emb2).sum(axis=1) # inner products
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
    scores = inn / emb1norm / emb2norm # cosine similarity
    return scores



###############################################################################
###                                 Main                                    ###
###############################################################################

params = params()
parr4para = {}
sarr4para = {}
scoring_function = weighted_average_sim_rmpc
wordfile = wordfiles[0]


### load word embeddings map
print 'loading word embeddings map...will take many minutes'
(words, We) = data_io.getWordmap(wordfile)
print 'word vectors loaded from %s' % wordfile
    

### tune weight and rmpc params in evaluation of similarity
for weightpara in weightparas:
    print '\n==================================\n'
    #### load weight for each word using idf or tf-idf
    word2weight = data_io.getWordWeight(weightfile, weightpara)
    print 'done with word2weight' 
    weight4ind = data_io.getWeight(words, word2weight)
    print 'done with weight4ind'
    
    for rmpc in rmpcs:
        print 'word vectors loaded from %s' % wordfile
        print 'word weights computed from %s using parameter a=%f' % (weightfile, weightpara)
        params.rmpc = rmpc
        print ' * remove the first %d principal components * ' % rmpc

        # for file in paraphrases_files:
        p, s = evaluate_similarity(We, words, paraphrases_file, weight4ind, scoring_function, params)
        print '\t  ' + str(paraphrases_file) + ': pearson: ' + str(p) + '; spearman: ' + str(s)
        for i in farr:
            p,s = evaluate_similarity(We, words, data_dir+i, weight4ind, scoring_function, params)
            print '\t' + str(i) + ': pearson: ' + str(p) + '; spearman: ' + str(s)

        ### save results
        if save_result:
            result_file = './result/sim_sif.result'
            comment4para = [ # need to align with the following loop
                ['word vector files', wordfiles], # comments and values,
                ['weight parameters', weightparas],
                ['remove principal component or not', rmpcs]
            ]
            with open(result_file, 'w') as f:
                pickle.dump([parr4para, sarr4para, comment4para] , f)


###############################################################################
###                                 Dead                                    ###
###############################################################################




