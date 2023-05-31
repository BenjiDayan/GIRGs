import sys
sys.path.append('../nemo-eva/src/')

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import networkit as nk
from tqdm import tqdm
import networkx as nx

from benji_girgs import generation, utils, plotting, fitting
import geopandas as gpd

import feature_extractor


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import feature_extractor
fe = feature_extractor.FeatureExtractor([])


# n=500
# d=2
# tau=2.5
# alpha=5.0
# desiredAvgDegree=20.0
# gnk, edges, weights, pts, const, id2gnk = generation.cgirg_gen(4*n, d, tau, alpha, desiredAvgDegree)
# pts = np.array(pts)
#
#
# # foo = fe.fit_ndgirg_general(2, utils.LCC, cube=True)
# #
# # info, g_out = foo(gnk)
#
# foo = fe.fit_ndgirg_binsearch(2)
#
# info, g_out = foo(gnk)

d = 2

g = nk.readGraph('/cluster/scratch/bdayan/GIRG_data/bio-dmela.SpaceOne', nk.Format.EdgeListSpaceOne)

info, g_out = fe.fit_ndgirg_general(d, utils.LCC, cube=True, verbose=True)(g)