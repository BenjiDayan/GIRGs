import os
os.chdir('/cluster/home/bdayan/girgs/benji_src/')
import sys
sys.path.append('../')
sys.path.append('../nemo-eva/src/')

import numpy as np
import pandas as pd

import networkit as nk
from benji_girgs import fitting, utils, generation, plotting

import seaborn as sns
import powerlaw
import matplotlib.pyplot as plt
import inspect
import powerlaw
import glob

from girg_sampling import girgs
import feature_extractor


if __name__ == '__main__':
    fe = feature_extractor.FeatureExtractor([])

    g_real = nk.readGraph('/cluster/scratch/bdayan/GIRG_data/bio-celegans.SpaceOne', nk.Format.EdgeListSpaceOne)
    info, g_out = fe.fit_ndgirg_binsearch(3)(g_real)
    print('\n'.join(info.split('|')))

