import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns

# Spot with horizontal layout
# first row goes from 0 to 11 (left to right)
spotsh = np.arange(48).reshape(4, 12)

# Spots with vertical layout
# The top left corner (0, 0) is 47, columns decrease as row increases 
spotsv = spotsh.T[::-1, ::-1]


def heatmap48(values=None, title=None, vert=False, figsize=(14, 4), ax=None, skip_ch=None, **kwargs):   
    if values is None:
        values = np.arange(48)
    values = np.asfarray(values)

    if values.shape == (48,):
        values = values.reshape(4, 12)
    elif values.shape == (12, 4):
        values = values.T[::-1, ::-1]
        
    if values.shape != (4, 12):
        raise ValueError('Input data has wrong shape (%s). If must be (48,), (12, 4) or (4, 12).' % values.shape)
    
    if vert:
        values = values.T[::-1, ::-1]
        figsize = figsize[::-1]
    
    if skip_ch is not None:
        mask = np.zeros(48, dtype=bool)
        for ch in skip_ch:
            mask[ch] = True
        kwargs['mask'] = mask.reshape(4, 12)
    
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
        