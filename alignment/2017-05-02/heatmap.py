import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns

# Spot with horizontal layout
# first row goes from 0 to 11 (left to right)
spotsh = np.arange(48).reshape(4, 12)

# Spots with vertical layout
# The top left corner (0, 0) is 47, columns decrease as row increases 
spotsv = spotsh.T[::-1, ::-1]


def heatmap48(values=None, title=None, vert=False, figsize=(14, 4), ax=None, nrows=4, ncols=12, **kwargs):   
    nspots = nrows*ncols
    if values is None:
        values = np.arange(nspots)
    values = np.asfarray(values)
    
    if values.shape == (nspots,):
        values = values.reshape(nrows, ncols)
    elif values.shape == (ncols, nrows):
        values = values.T[::-1, ::-1]
        
    if values.shape != (nrows, ncols):
        raise ValueError('Input data has wrong shape ({shape}). If must be ({nspot},), '
                         '({ncols}, {nrows}) or ({nrows}, ncols).'
                         .format(shape=values.shape, nrows=nrows, ncols=ncols, nspot=nspot))
    
    if vert:
        values = values.T[::-1, ::-1]
        figsize = figsize[::-1]
    
    default_style = dict(cmap='viridis', fmt='.0f', cbar_kws=dict(aspect=15))
    for k, v in default_style.items(): 
        kwargs.setdefault(k, v)
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(values, **kwargs)
    if title is not None:
        ax.set_title(title, va='bottom')


def annonotate_spots(ax=None, color='w', weight='heavy', **kws):
    if ax is None:
        ax = plt.gca()
    vert = False if ax.get_xlim()[1] > ax.get_ylim()[1] else True
    for i in spotsv.ravel():
        x, y = np.where(spotsv == i)
        x = 11 - x
        if vert:
            x, y = y, x
        ax.text(x + 0.5, y + 0.5, i, va='center', ha='center', color='w', fontdict=dict(weight='heavy'), **kws)
        