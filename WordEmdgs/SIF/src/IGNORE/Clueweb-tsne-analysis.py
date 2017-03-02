
# coding: utf-8

# In[1]:

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
import matplotlib.cm as cm
import matplotlib.patches as mpatches



    
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

## run
### the following is the word embeddings maps to use, download them pretrained
wordfiles = [#'../data/paragram_sl999_small.txt', # need to download it from John Wieting's github (https://github.com/jwieting/iclr2016)
    # '/Users/corbinrosset/Dropbox/GloVe/glove.840B.300d.txt'
    '/Users/corbinrosset/Dropbox/Paragrams/paragrams-XXL-SL999.txt'

    ]
weightfile = '/Users/corbinrosset/Dropbox/Arora/QA-code/src/WordEmdgs/SIF/auxiliary_data/enwiki_vocab_min200.txt'
weightparas = [1e-2] ### used 1e-2 #[-1,1e-1,1e-2,1e-3,1e-4] # (0.01 works best)
# weightparas = [-1,1e-1,1e-2,1e-3,1e-4] # (0.01 works best)

rmpcs = [0] #[0, 1, 2, 3, 4] # remove the first k principle components, (4 is best)
scoring_function = None
save_result = False 

data_dir = '/Users/corbinrosset/Dropbox/Arora/QA-data/VanDurme_FB_annotations/annotated_clueweb/ClueWeb09_English_1/processed/' ### clueweb data
# data_dir = '../data/' ### for textual similarity data
paraphrases_file = 'grouped_textual_triples.pkl' ### for clueweb data
# paraphrases_file = 'combined_test_data_normed' ### for textual similarity data

data = data_dir + paraphrases_file

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


execute_path = '/Users/corbinrosset/Dropbox/Arora/QA-code/'
params = params()
parr4para = {}
sarr4para = {}
wordfile = wordfiles[0]




# In[2]:

### load word embeddings map
print 'loading word embeddings map...will take many minutes'
start = time.clock() 
(words, We) = data_io.getWordmap(wordfile)
elapsed = time.clock()
elapsed = elapsed - start
print 'word vectors loaded from %s' % wordfile
print "loaded word vectors in: ", elapsed
    


# In[3]:

### load data
print 'loading data'
start = time.clock() 
DATA = pickle.load(open(data, 'r'))
elapsed = time.clock()
elapsed = elapsed - start
print "loaded ", str(data), " data in: ", elapsed


# In[8]:


###############################################################################
###            Helper Methods for the Original Textual Similarity Task      ###
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

def get_embeddings(We, words, data, weight4ind, params):

    ### asssuming data is the similarity dataset:
    seq, labels, sentences = [], [], []
    with open(data,'r') as fl:
        for idx, i in enumerate(fl):
            i = i.split("\t")
            p1 = i[0]; p2 = i[1]; score = float(i[2])
            X1, X2 = data_io.getSeqs(p1,p2,words)
            seq.append(X1)
            seq.append(X2)
            labels.append(score)
            labels.append(score)
            sentences.append(p1)
            sentences.append(p2)

    x1, w1 = data_io.prepare_data(seq)
    emb, embnorm = get_norm_embedding(We, x1, w1, params)
    emb = np.asarray(emb, dtype=np.float64)
    # perhaps divide each row by its L2 norm, emb /= embnorm
    print np.shape(emb)
    return emb, labels, sentences




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
    '''basically does what the scoring function below does, but for one sentence'''
    emb1 = get_weighted_average(We, x1, w1) # num samples by embedding dim
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







# In[ ]:

###############################################################################
###                 Helper Methods for Clueweb-related tasks                ###
###############################################################################

### CLUEWEB
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

### CLUEWEB
def get_embeddings_clueweb(We, words, data, weight4ind, params, key):
    seq, labels, sentences = [], [], []
    key = str(key[0][0]) + '-' + key[1][0] + '-' + str(key[2][0])
    for i, text_instance in enumerate(data):
        ### remember text instances are tuples ("text goes here",)
        X = data_io.getSeq(text_instance[0].strip(), words)
        labels.append(key)
        seq.append(X)
        sentences.append(text_instance)

    x1, w1 = data_io.prepare_data(seq)
    emb, embnorm = get_norm_embedding(We, x1, w1, params)
    emb = np.asarray(emb, dtype=np.float64)
    # perhaps divide each row by its L2 norm, emb /= embnorm
    return emb, labels, sentences

### CLUEWEB
def evaluate_similarity_all_pairs(We, words, data, weight4ind, scoring_function, params):
    '''data here is a dictionary of 5k of the most commonly mentioned freebase triples 
        mapping to a list of their textual instances, which is at most 40k in length. 
        The goal here is to get a sense of how similar the instances in the list are with 
        each other - there are no labels here, just output say the average/median 
        similarity score of all pairs in the list, or a subsample of all pairs. 
    '''
    triple_sim_stats = {}
    avgs = []
    prob = 0.001 
    counter = 0
    seq1, seq2 = [], []
    keys = sorted(data, key=lambda x: len(data[x]), reverse=True)
    try:
        output = open('output_similarity_all_pairs.txt', 'w')
    except:
        print 'failure'
        exit()
        
    for key in keys:
        texts = data[key]
        pairs = itertools.combinations(texts, 2)
        counter = 10000
        seq1, seq2 = [], []
        output.write('-------------------\n')
        output.write(str(key) + '\n')
        output.write('number of textual instances: ' + str(len(texts)) + '\n')

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
        output.write('length of sampled sequence: ' + str(len(seq1)) + '\n')
        assert len(seq1) == len(seq2)

        if len(seq1) == 0:
            output.write('skipped this triple!\n')
            continue
        x1,m1 = data_io.prepare_data(seq1)
        x2,m2 = data_io.prepare_data(seq2)
        m1 = data_io.seq2weight(x1, m1, weight4ind)
        m2 = data_io.seq2weight(x2, m2, weight4ind)
        scores = scoring_function(We,x1,x2,m1,m2, params)
        preds = np.squeeze(scores)
        stats = get_statistics(preds)
        output.write(pprint.pformat(stats) + '\n')
        triple_sim_stats[key] = stats
        avgs.append(stats['mean'])
        print 'done with ' + str(key[0][0]) + '-' + key[1][0] + '-' + str(key[2][0])
    output.close()
    return triple_sim_stats, avgs


# In[ ]:

# %matplotlib inline
# mpld3.enable_notebook()
mpld3.disable_notebook()

###############################################################################
###                 Manage and Visualize Clueweb Experiments                ###
###############################################################################

def t_sne(data):
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

    X_2d = bh_sne(data, perplexity=30, theta=0.5) #, max_iter=600) #bh_sne(X, perplexity=50, theta=0.5)  
    return X_2d

def tooltip_style(text):
    return "<p style=\"color: #ffffff; background-color: #000000\">" + str(text) + "</p>"

def plot_clueweb(X, Y, popup_labels, title):
    '''X is a list of matrices of length n, each of which has its own class or 
    color, y. popup_labels is also a list of n lists, each containing the tag
    of individual points (rows in a matrix of X)'''

    ###############
    labels_flat = [item for sublist in Y for item in sublist]
    labels_set = set(labels_flat)
            
    fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'), figsize=(20,20))
    
    assert len(X) == len(Y)
    if popup_labels:
        assert len(popup_labels) == len(X)
        assert np.shape(X)[0] == np.shape(popup_labels)[0]
    ax.grid(color='white', linestyle='solid')
    colors = cm.rainbow(np.linspace(0,1, len(Y)))
    color_label_map = {}
    for c, label in zip(colors, labels_set):
        color_label_map[label] = c
#     print 'labels\' colors: ' + str(color_label_map)
        
    legend_partition, legend_strings = [], [] 
    for (x, group_labels, tag) in zip(X, Y, popup_labels):
        label = group_labels[0]
        assert all(label == i for i in group_labels)
        scatter = ax.scatter(x[:, 0], x[:, 1], c = color_label_map[label], alpha=0.3, label=label)
        legend_partition.append(scatter)
        legend_strings.append(label)
#         tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=[tooltip_style(t[0]) for t in tag])
        tooltip = mpld3.plugins.PointHTMLTooltip(scatter, labels=[tooltip_style(t[0]) for t in tag])
        mpld3.plugins.connect(fig, tooltip)
        
    l_keys, l_nums, recs = [], [], []
    print 'Legend: '
    for i, (key, value) in enumerate(color_label_map.items()):
        print str(i) + ' = ' + str(key)
        l_nums.append(str(i))
        l_keys.append(key)
        recs.append(mpatches.Rectangle((0,0),1,1,fc=value))

    plt.legend(recs, l_keys, loc='upper right', fontsize=8)
    if title:
        ax.set_title(title, size=12)
        
    mpld3.save_html(fig, title + '.html')
    mpld3.show()
#     mpld3.display() # doesn't seem to work for ipython notebooks?
    return

def clueweb_tsne_experiment():
     ### T-SNE on selected triples - A COUPLE OF EXPERIMENTS NEED TO BE DONE HERE!!!!
    with open('./common_base_triples_3.txt') as g:
        ### only load textual mentions of triples with the most mentions
        keys = [eval(i.strip()) for i in g.readlines()]
        X, labels, sentences = [], [], []
        for key in keys:
            data_selected = DATA[key]
            x, labels_x, sentences_x = get_embeddings_clueweb(We, words, data_selected, weight4ind, params, key)
            if np.shape(x)[0] > 7000:  ### sample 5000 instances to reduce runtime
                idx = np.random.choice(np.shape(x)[0], 7000)
                X.append(x[idx, :])
                labels.append((np.array(labels_x)[idx]).tolist())
                sentences.append((np.array(sentences_x)[idx]).tolist())
            else:
                X.append(x)
                labels.append(labels_x)
                sentences.append(sentences_x)
            print '\t' + str(np.shape(X[-1])) + ' ' + str(key[0][0]) + '-' + key[1][0] + '-' + str(key[2][0])
        print 'done creating embeddings'
        X_stacked = np.asarray(np.vstack(tuple(X)), dtype=np.float64)
        X_2d = t_sne(X_stacked) #
        print 'done with tsne ' + str(np.shape(X_2d)) 

        ### must chunk the t-sne embeddings to match labels...
        X_2d_chunk = []
        size = 0
        for i in labels:
            X_2d_chunk.append(X_2d[size:(size + len(i))])
            size += len(i)

        for a, b, c in zip(X_2d_chunk, labels, sentences):
            assert len(a) == len(b) == len(c)
        return X_2d_chunk, labels, sentences


# In[10]:

###############################################################################
###                                 Main                                    ###
###############################################################################


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
        print '*** remove the first %d principal components ***' % rmpc

        #################################################################
        ### modify the following lines for your purposes (clueweb or sim)
        #################################################################


        #################################################################
        #################################################################
        ### for text similarity task data:

        # for file in paraphrases_files:
        # scoring_function = weighted_average_sim_rmpc
        # triple_sim_stats, avgs = evaluate_similarity(We, words, data, weight4ind, scoring_function, params) ### only for clueweb

        # print 'average similarities per triple: ' + str(avgs)
        # print 'average of averages: ' + str(np.mean(avgs))
        # with open(data_dir + 'clueweb_triples_similarity_' + str(rmpc) + '_rmpc_' + str(weightpara) + '_weight.pkl', 'w') as f:
        #     pickle.dump(triple_sim_stats, f)

        #################################################################
        ### T-SNE
        # X, labels, sentences = get_embeddings(We, words, data, weight4ind, params)
        # t_sne(X, labels, sentences, 't-SNE on text similarity pairs')

        #################################################################
        #################################################################
        ### for clueweb data

        ### test similirity of textual instances of same FB triple...
#         scoring_function = weighted_average_sim_rmpc
#         triple_sim_stats, avgs = evaluate_similarity_all_pairs(We, words, DATA, weight4ind, scoring_function, params) ### only for clueweb
#         print 'average similarities per triple: ' + str(avgs)
#         print 'average of averages: ' + str(np.mean(avgs))
#         with open(data_dir + 'clueweb_triples_similarity_' + str(rmpc) + '_rmpc_' + str(weightpara) + '_weight.pkl', 'w') as f:
#             pickle.dump(triple_sim_stats, f)

        #################################################################
#         X_2d_chunk, labels, sentences = clueweb_tsne_experiment()
#         plot_clueweb(X_2d_chunk, labels, sentences, title='t-SNE of textual instances of most common clueweb triples (6) (sampled)')
#         print 'done'
       
            
            



###############################################################################
###                                 Dead                                    ###
###############################################################################


# In[ ]:



