import pickle, sys
sys.path.append('../src')
import data_io # eval
import random
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pprint
from scipy.stats import iqr
import time
import random
import itertools
import gzip, cPickle
import matplotlib.pyplot as plt
from tsne import bh_sne
import mpld3



## run
### the following is the word embeddings maps to use, download them pretrained
wordfiles = [#'../data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
    # '/Users/corbinrosset/Dropbox/GloVe/glove.840B.300d.txt'
    '/Users/corbinrosset/Dropbox/Paragrams/paragrams-XXL-SL999.txt'

    ]
weightfile = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/WordEmdgs/SIF/auxiliary_data/enwiki_vocab_min200.txt'
weightparas = [1e-2] ### used 1e-2 #[-1,1e-1,1e-2,1e-3,1e-4] # (0.01 works best)
# weightparas = [-1,1e-1,1e-2,1e-3,1e-4] # (0.01 works best)

rmpcs = [3] #[0, 1, 2, 3, 4] # remove the first k principle components, (4 is best)
scoring_function = None
save_result = False 

# data_dir = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/' 
data_dir = '../data/'
# paraphrases_file = 'grouped_textual_triples.pkl' 
paraphrases_file = 'combined_test_data_normed' ### data to evaluate
data = data_dir + paraphrases_file
tsne_name = './tnse_similarity.html'

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

def plot(X, y, popup_labels=None):

    fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
    N = np.shape(X)[0]

    scatter = ax.scatter(X[:, 0],
                         X[:, 1],
                         c=y,
                         alpha=0.5)
    ax.grid(color='white', linestyle='solid')

    ax.set_title("Scatter Plot (with tooltips!)", size=20)

    if not popup_labels:
        labels = ['point {0}'.format(i + 1) for i in range(N)]
    else:
        labels = popup_labels
    
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    mpld3.plugins.connect(fig, tooltip)
    mpld3.save_html(fig, tsne_name)
    mpld3.show()
    return

def t_sne(data, labels, sentences):
    '''run_bh_tsne(data, no_dims=2, perplexity=50, theta=0.5, randseed=-1, verbose=False,initial_dims=50, use_pca=True, max_iter=1000):
        
        Run TSNE based on the Barnes-HT algorithm

        Parameters:
        ----------
        data: file or numpy.array
            The data used to run TSNE, one sample per row
        no_dims: int
        perplexity: int
        randseed: int
        theta: float
        initial_dims: int
        verbose: boolean
        use_pca: boolean
        max_iter: int
    '''

    X_2d = bh_sne(data) #bh_sne(X, perplexity=50, theta=0.5)  
    print np.shape(X_2d)
    plot(X_2d, labels, sentences)

def get_statistics(lst):
    stats = {}
    stats['mean'] = np.mean(lst)
    stats['median'] = np.median(lst)
    # stats['quartiles'] = 
    stats['standard deviation'] = np.std(lst)
    stats['mean absolute deviation'] = np.mean(np.absolute(lst - np.mean(lst)))
    stats['interquartile range'] = iqr(lst)

    # plt.clera()
    # plt.hist(lst, bins=100)
    # plt.show()

    return stats

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

def get_embeddings(We, words, data, weight4ind, params):

    ### asssuming data is the similarity dataset:
    seq, labels, sentences = [], [], []
    with open(f,'r') as fl:
        for idx, i in enumerate(fl):
            i = i.split("\t")
            p1 = i[0]; p2 = i[1]; score = float(i[2])
            X1, X2 = data_io.getSeqs(p1,p2,words)
            seq.append(X1).append(X2)
            labels.append(score).append(score)
            sentences.append(p1).append(p2)

    x1,m1 = data_io.prepare_data(seq)
    return x1, labels, sentences

    ### if data is the clueweb dataset
    # for key in target_triples:
    #     texts = data[key]
    #     seq1= []
    #     print '-------------------\n'
    #     print key
    #     print 'number of textual instances: ' + str(len(texts))

    #     for i, text_instance in enumerate(data[key]):
    #         X = data_io.getSeq(text_instance.strip())
    #         seq1.append(X)

    #     if len(seq1) == 0:
    #         print 'skipped'
    #         continue
    #     x1, m1 = data_io.prepare_data(seq1)
    #     m1 = data_io.seq2weight(x1, m1, weight4ind)
    #     scores = scoring_function(We,x1,x2,m1,m2, params)
    #     preds = np.squeeze(scores)
    #     stats = get_statistics(preds)
    #     pprint.pprint(stats)
    #     triple_sim_stats[key] = stats
    #     avgs.append(stats['mean'])
    # return triple_sim_stats, avgs


    # scipy.io.savemat('output_word_embeddings', mdict, appendmat=True)


def evaluate_similarity_all_pairs(We, words, data, weight4ind, scoring_function, params):
    '''data here is a dictionary of 5k of the most commonly mentioned freebase triples mapping to a list of their textual instances, which is at most 40k in length. The goal here is to get a sense of how similar the instances in the list are with each other - there are no labels here, just output say the average/median similarity score of all pairs in the list, or a subsample of all pairs. 
    '''
    triple_sim_stats = {}
    avgs = []
    prob = 0.001 
    counter = 0
    seq1, seq2 = [], []

    ### load textual mentions of triples with the most mentions
    start = time.clock() 
    data = pickle.load(open(data, 'r'))
    elapsed = time.clock()
    elapsed = elapsed - start
    print "loaded textual mention data in: ", elapsed
    keys = sorted(data, key=lambda x: len(data[x]), reverse=True)
    
    for key in keys:
        texts = data[key]
        pairs = itertools.combinations(texts, 2)
        counter = 10000
        seq1, seq2 = [], []
        print '-------------------\n'
        print key
        print 'number of textual instances: ' + str(len(texts))

        for i, pair in enumerate(pairs):
            if random.random() > prob: ### subsample
                continue
            counter -= 1
            if counter <= 0:
                break
            pair = (pair[0][0].strip(), pair[1][0].strip())
            X1, X2 = data_io.getSeqs(pair[0],pair[1], words)
            seq1.append(X1)
            seq2.append(X2)
        print 'length of sampled sequence: ' + str(len(seq1))
        assert len(seq1) == len(seq2)

        if len(seq1) == 0:
            print 'skipped'
            continue
        x1,m1 = data_io.prepare_data(seq1)
        x2,m2 = data_io.prepare_data(seq2)
        m1 = data_io.seq2weight(x1, m1, weight4ind)
        m2 = data_io.seq2weight(x2, m2, weight4ind)
        scores = scoring_function(We,x1,x2,m1,m2, params)
        preds = np.squeeze(scores)
        stats = get_statistics(preds)
        pprint.pprint(stats)
        triple_sim_stats[key] = stats
        avgs.append(stats['mean'])
    return triple_sim_stats, avgs


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

def get_norm_embedding(We, x1, w1, params):
    '''basically does what the scoring function below does'''
    emb1 = get_weighted_average(We, x1, w1)
    if  params.rmpc > 0:
        emb1 = remove_pc(emb1, params.rmpc)
    emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
    return emb1, emb1norm

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

execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/'
params = params()
parr4para = {}
sarr4para = {}
scoring_function = weighted_average_sim_rmpc
wordfile = wordfiles[0]

### load word embeddings map
print 'loading word embeddings map...will take many minutes'
start = time.clock() 
(words, We) = data_io.getWordmap(wordfile)
elapsed = time.clock()
elapsed = elapsed - start
print "loaded word vectors in: ", elapsed
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

        #################################################################
        ### modify the following lines for your purposes (clueweb or sim)
        #################################################################

        # for file in paraphrases_files:
        # triple_sim_stats, avgs = evaluate_similarity_all_pairs(We, words, data, weight4ind, scoring_function, params) ### only for clueweb

        # print 'average similarities per triple: ' + str(avgs)
        # print 'average of averages: ' + str(np.mean(avgs))
        # with open(data_dir + 'clueweb_triples_similarity_' + str(rmpc) + '_rmpc_' + str(weightpara) + '_weight.pkl', 'w') as f:
        #     pickle.dump(triple_sim_stats, f)

        X, labels, sentences = get_embeddings(We, words, data, weight4ind, params)
        t_sne(X, labels, sentences)
        print 'done'



###############################################################################
###                                 Dead                                    ###
###############################################################################




