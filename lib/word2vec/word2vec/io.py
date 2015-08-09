from __future__ import unicode_literals

from wordvectors import WordVectors
from wordclusters import WordClusters

import numpy as np


def load(fname, kind='auto', *args, **kwargs):
    '''
    Loads a word vectors file
    '''
    if kind == 'auto':
        if fname.endswith('.bin'):
            kind = 'bin'
        elif fname.endswith('.txt'):
            kind = 'txt'
        else:
            raise Exception('Could not identify kind')
    if kind == 'bin':
        return WordVectors.from_binary(fname, *args, **kwargs)
    elif kind == 'txt':
        return WordVectors.from_text(fname, *args, **kwargs)
    elif kind == 'mmap':
        return WordVectors.from_mmap(fname, *args, **kwargs)
    else:
        raise Exception('Unknown kind')


def load_clusters(fname):
    '''
    Loads a word cluster file
    '''
    return WordClusters.from_text(fname)
