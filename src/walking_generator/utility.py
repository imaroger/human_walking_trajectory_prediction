import numpy
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

def cast_array_as_matrix(array):
    """
    cast a numpy array as (size,1)-matrix to enable straight forward linear algebra
    """
    err_str = "Only accepts 1 dimensional numpy arrays as inputs.\n"
    assert len(array.shape) == 1, err_str
    return numpy.matrix(array.reshape(array.size, 1))

def color_matrix(M, ax=None, fig=None, title=None, cmap=cm.jet):
    """uses a blue to red cmap for matrix visualization"""
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    if not ax:
        ax = fig.add_subplot(1, 1, 1)

    if len(M.shape) == 1:
        dummy = M.reshape((M.shape[0], 1))

    else:
        dummy = M

    x = numpy.arange(dummy.shape[1])
    y = numpy.arange(dummy.shape[0])

    X,Y = numpy.meshgrid(x,y)

    mat = ax.matshow(dummy,
            cmap=cmap, vmax=abs(dummy).max(), vmin=-abs(dummy).max())
    #fig.colorbar(mat, ax=ax, shrink=0.9)
    if title:
        ax.set_title(title)
    ax.locator_params(tight=True)
