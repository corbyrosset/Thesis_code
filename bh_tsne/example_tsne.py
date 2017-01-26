import numpy as np
import gzip, cPickle
import matplotlib.pyplot as plt
from tsne import bh_sne
import mpld3


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
    mpld3.show()
    return


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

f = gzip.open("mnist.pkl.gz", "rb")
train, val, test = cPickle.load(f)
f.close()

X = np.array(test[0], dtype=np.float64)
y = test[1]
print np.shape(X)
X_2d = bh_sne(X) #bh_sne(X, perplexity=50, theta=0.5)  
print type(X_2d)
print np.shape(X_2d)

plot(X_2d, y)
