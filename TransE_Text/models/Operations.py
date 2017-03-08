import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
# from theano.tensor.signal.downsample import max_pool_2
from collections import OrderedDict


# Similarity functions -------------------------------------------------------
def L1sim(left, right):
    return - T.sum(T.abs_(left - right), axis=1)


def L2sim(left, right):
    return - T.sqrt(T.sum(T.sqr(left - right), axis=1))


def Dotsim(left, right):
    return T.sum(left * right, axis=1)
# -----------------------------------------------------------------------------


# Cost ------------------------------------------------------------------------
def margincost(pos, neg, marge=1.0):
    # assert np.all(T.ge(neg, 0.0))
    # assert np.all(T.ge(pos, 0.0))
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0 ### returns when margin violated

def squared_margin_cost(pos, neg, marge=1.0, magnifier=10.0):
    out = neg - pos + marge
    return T.sum(magnifier * out * out * (out > 0)), out > 0 

def exp_margin_cost(pos, neg, marge=1.0, magnifier=5.0):
    out = neg - pos + marge
    loss = T.exp(magnifier * out)
    
    # loss_pos = T.log(1 + T.exp(2*(marg_pos - pos)))
    # loss_neg = T.log(1 + T.exp(2*(neg - marg_neg)))
    # Dos Santos used marg_neg + neg instead of marg_neg - neg. That means
    # that, if say marg_neg = 0.5, then negative triples need to score below -0
    # .5 in order to incur zero loss. However, the scoring/similarity function 
    # is strictly nonnegative, so we have to modify marg_neg to be positive, 
    # and put a minus sign.
    return T.sum(loss * (out > 0.0)), out > 0.0 # returns when margin violated
# -----------------------------------------------------------------------------


# Activation functions --------------------------------------------------------
def rect(x):
    return x * (x > 0)


def sigm(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def lin(x):
    return x
# -----------------------------------------------------------------------------

# Optimizers ------------------------------------------------------------------
# All optimizers return an ordered dictionary of updates to parameters, in the 
# order they were supplied in in "all_params"

def sgd(loss, all_params, learning_rate=0.01):
    gradients = T.grad(loss, all_params)
    updates = OrderedDict((i, i - learning_rate * j) for i, j in zip(all_params, gradients))
    return updates

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = OrderedDict()
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    #(Decay the first moment running average coefficient)
    b1_t = b1*gamma**(t-1)  

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        # (Update biased first moment estimate)
        m = b1_t*m_previous + (1 - b1_t)*g 
        # (Update biased second raw moment estimate)
        v = b2*v_previous + (1 - b2)*g**2      
        # (Compute bias-corrected first moment estimate)             
        m_hat = m / (1-b1**t)
        # (Compute bias-corrected second raw moment estimate                       
        v_hat = v / (1-b2**t)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) 
        
        # Update parameters
        # updates[m_previous] = m
        # updates[v_previous] = v
        updates[theta_previous] = theta
    # updates.append((t, t + 1.))
    return updates

def RMSprop(loss, all_params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(loss, all_params)
    updates = OrderedDict()
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        # updates.append((acc, acc_new))
        # updates[acc] = acc_new
        # updates.append((p, p - lr * g))
        updates[p] = (p - lr * g)

    return updates
# -----------------------------------------------------------------------------


# Layers ----------------------------------------------------------------------
class Layer(object):
    """Class for a layer with one input vector w/o biases."""

    def __init__(self, rng, act, n_inp, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inp: input dimension.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        wbound = np.sqrt(6. / (n_inp + n_out))
        W_values = np.asarray(
            rng.uniform(low=-wbound, high=wbound, size=(n_inp, n_out)),
            dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W' + tag)
        self.params = [self.W]

    def __call__(self, x):
        """Forward function."""
        return self.act(T.dot(x, self.W))


class LayerLinear(object):
    """Class for a layer with two inputs vectors with biases."""

    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inpl: dimension of the 'left' input.
        :param n_inpr: dimension of the 'right' input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = eval(act)
        self.actstr = act
        self.n_inpl = n_inpl
        self.n_inpr = n_inpr
        self.n_out = n_out
        self.layerl = Layer(rng, 'lin', n_inpl, n_out, tag='left' + tag)
        self.layerr = Layer(rng, 'lin', n_inpr, n_out, tag='right' + tag)
        b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b' + tag)
        self.params = self.layerl.params + self.layerr.params + [self.b]

    def __call__(self, x, y):
        """Forward function."""
        return self.act(self.layerl(x) + self.layerr(y) + self.b)


class LayerBilinear(object):
    """
    Class for a layer with bilinear interaction (n-mode vector-tensor product)
    on two input vectors with a tensor of parameters.
    """

    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inpl: dimension of the 'left' input.
        :param n_inpr: dimension of the 'right' input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = eval(act)
        self.actstr = act
        self.n_inpl = n_inpl
        self.n_inpr = n_inpr
        self.n_out = n_out
        wbound = np.sqrt(9. / (n_inpl + n_inpr + n_out))
        W_values = rng.uniform(low=-wbound, high=wbound,
                               size=(n_inpl, n_inpr, n_out))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W' + tag)
        b_values = np.zeros((n_out, ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b' + tag)
        self.params = [self.W, self.b]

    def __call__(self, x, y):
        """Forward function."""
        xW = T.tensordot(x, self.W, axes=([1], [0]))
        xWy = ((y.reshape((y.shape[0], y.shape[1], 1))) * xW).sum(1)
        return self.act(xWy + self.b)

    def forwardrankrel(self, x, y):
        """Forward function in the special case of relation ranking to avoid a
        broadcast problem. @TODO: think about a workaround."""
        xW = T.tensordot(x, self.W, axes=([1], [0]))
        xW = xW.reshape((1, xW.shape[1], xW.shape[2]))
        xWy = ((y.reshape((y.shape[0], y.shape[1], 1))) * xW).sum(1)
        return self.act(xWy + self.b)


class LayerMat(object):
    """
    Class for a layer with two input vectors, the 'right' member being a flat
    representation of a matrix on which to perform the dot product with the
    'left' vector [Structured Embeddings model, Bordes et al. AAAI 2011].
    """

    def __init__(self, act, n_inp, n_out):
        """
        Constructor.

        :param act: name of the activation function ('lin', 'rect', 'tanh' or
                    'sigm').
        :param n_inp: input dimension.
        :param n_out: output dimension.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.act = eval(act)
        self.actstr = act
        self.n_inp = n_inp
        self.n_out = n_out
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.
        ry = y.reshape((y.shape[0], self.n_inp, self.n_out))
        rx = x.reshape((x.shape[0], x.shape[1], 1))
        return self.act((rx * ry).sum(1))


class LayerTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of 
    of the 'left member' and 'right member'i.e. translating x by y.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x+y

class LayerdMat(object):
    """
    
    """

    def __init__(self):
        """
        Constructor.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.

        return x * y

class Unstructured(object):
    """
    Class for a layer with two input vectors that performs the linear operator
    of the 'left member'.

    :note: The 'right' member is the relation, therefore this class allows to
    define an unstructured layer (no effect of the relation) in the same
    framework.
    """

    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x

# ----------------------------------------------------------------------------

# Embeddings class -----------------------------------------------------------
class Embeddings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D
        wbound = np.sqrt(6. / D)
        W_values = rng.uniform(low=-wbound, high=wbound, size=(D, N))
        W_values = W_values / np.sqrt(np.sum(W_values ** 2, axis=0))
        W_values = np.asarray(W_values, dtype=theano.config.floatX)
        self.E = theano.shared(value=W_values, name='E' + tag)
        # Define a normalization function with respect to the L_2 norm of the
        # embedding vectors.
        self.updates = OrderedDict({self.E: self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)

        self.L1_norm = T.sum(abs(self.E))
        self.L2_sqr_norm = T.sum(self.E ** 2)

class WordEmbeddings(object):

    def __init__(self, vocabSize, dim, wordFile=None, vocab=None):
        '''
        :param vocabSize: the number of words in the vocabulary
        :param dim: the dimension of each word embedding
        :param wordFile: the word embedding to initialize, else random
        :param vocab: the words in the vocabulary, plus an UNK token

        if a dictionary is provided, project the word embeddings onto 
        the dictionary. We want take the intersection of the provided 
        dictionary words and the word embedddings. 
        '''
        self.vocabSize = 0
        self.dim = dim
        self.wordFile = wordFile
        self.vocab = vocab
        if self.vocab: # if a valid dictionary file is provided
            dictionary = set([])
            for i in open(self.vocab, 'r'):
                word = i.strip().lower()
                if word:
                    dictionary.add(i.strip().lower())
            dictionary.add('UUUNKKK') ### OOV words, very important
            self.vocab = dictionary
            print 'loaded vocabulary of %s words' % len(self.vocab)

        if self.wordFile:
            print 'loading word embeddings from %s' % wordFile
            # map words into their index (row) of word embedding matrix 
            self.vocab2indx = {} 
            self.We = [] # word embedding matrix
            counter = 0
            with open(self.wordFile,'r') as f:
                for (idx, line) in enumerate(f):
                    line = line.split()
                    word = line[0].decode('utf8','ignore')
                    vec = line[1:]
                    if self.vocab and word not in self.vocab:
                        ### not in dictionary
                        continue
                    if word in self.vocab2indx:
                        ### this is ok, since the file is actually a 
                        # concatenation of word2vec and paragrams, so there
                        # will be some repeat word embeddings
                        continue
                    assert len(vec) == dim
                    self.vocab2indx[word] = counter
                    self.We.append(vec)
                    counter += 1
            if 'UUUNKKK' not in self.vocab2indx:
                self.vocab2indx['UUUNKKK'] = counter
                self.We.append([0.01] * dim)
                counter += 1

            temp = np.asarray(self.We, dtype=theano.config.floatX)
            self.We = theano.shared(value=temp, name='WordEmbedding')
            self.updates = OrderedDict({self.We: self.We / T.sqrt(T.sum(self.We ** 2, axis=0))})
            self.normalize = theano.function([], [], updates=self.updates)
            print 'initialized %s word embeddings, out of %s that were provided in file' % (str(np.shape(temp)), idx)
            # vocab indices must match word embedding indices
            self.vocabSize = len(self.vocab2indx)
            assert len(self.vocab2indx) == np.shape(temp)[0] 

        else:
            if vocab is None:
                print 'if a wordfile is not supplied, you must supply a vocab'
                exit()
            raise NotImplementedError
        self.L1_norm = T.sum(abs(self.We))
        self.L2_sqr_norm = T.sum(self.We ** 2)


    def getEmbeddings(self):
        return self.We

    def getVocab2Idx(self):
        return self.vocab2indx

    def getNormalizeFn(self):
        return self.normalize

### from https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py
# class CNN:
#     def floatX(X):
#         return np.asarray(X, dtype=theano.config.floatX)

#     def init_weights(shape):
#         return theano.shared(floatX(np.random.randn(*shape) * 0.01))

#     def rectify(X):
#         return T.maximum(X, 0.)

#     def softmax(X):
#         e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
#         return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

#     def dropout(X, p=0.):
#         if p > 0:
#             retain_prob = 1 - p
#             X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
#             X /= retain_prob
#         return X

#     def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
#         l1a = rectify(conv2d(X, w, border_mode='full'))
#         l1 = max_pool_2d(l1a, (2, 2))
#         l1 = dropout(l1, p_drop_conv)

#         l2a = rectify(conv2d(l1, w2))
#         l2 = max_pool_2d(l2a, (2, 2))
#         l2 = dropout(l2, p_drop_conv)

#         l3a = rectify(conv2d(l2, w3))
#         l3b = max_pool_2d(l3a, (2, 2))
#         l3 = T.flatten(l3b, outdim=2)
#         l3 = dropout(l3, p_drop_conv)

#         l4 = rectify(T.dot(l3, w4))
#         l4 = dropout(l4, p_drop_hidden)

#         pyx = softmax(T.dot(l4, w_o))
#         return l1, l2, l3, l4, pyx

#     trX, teX, trY, teY = mnist(onehot=True)

#     trX = trX.reshape(-1, 1, 28, 28)
#     teX = teX.reshape(-1, 1, 28, 28)

#     X = T.ftensor4()
#     Y = T.fmatrix()

#     w = init_weights((32, 1, 3, 3))
#     w2 = init_weights((64, 32, 3, 3))
#     w3 = init_weights((128, 64, 3, 3))
#     w4 = init_weights((128 * 3 * 3, 625))
#     w_o = init_weights((625, 10))

#     noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
#     l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
#     y_x = T.argmax(py_x, axis=1)

    # cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
    # params = [w, w2, w3, w4, w_o]
    # updates = RMSprop(cost, params, lr=0.001)

    # train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    # predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

