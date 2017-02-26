import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
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
    out = neg - pos + marge
    return T.sum(out * (out > 0)), out > 0 ### returns when margin violated
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

