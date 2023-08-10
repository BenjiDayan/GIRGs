import os
os.chdir('/cluster/home/bdayan/girgs/benji_src/')

from benji_girgs import utils, generation, points, fitting, mcmc
import glob
import pandas as pd
import networkit

import networkit as nk


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import numpy as np
import matplotlib.pyplot as plt

import do_feature_extract
import networkx as nx

import pickle


n=1000
tau=2.5
alpha=1.5
desiredAvgDegree=50.0
d=2

g, edges, weights, pts, const = generation.generate_GIRG_nk(
        n, d, tau, alpha,
    desiredAvgDegree=desiredAvgDegree,
    points_type=points.PointsTorus2, c_implementation=True)


a, B, pts_dm = utils.get_diffmap_and_points(g, 10, 2, process=None)

gnx = nk.nxadapter.nk2nx(g)
A = nx.linalg.adjacency_matrix(gnx).todense()
D = np.array([x[1] for x in (gnx.degree)])
D_h = D**(0.5)
D_hi = D**(-0.5)

Iweighting=0.5
M = np.diag(1/D) @ A

######
gamma=0.9
M = M @ np.diag(D**(-gamma))
M = np.diag(np.array(1 / M.sum(axis=-1)).squeeze()) @ M
######

M = (1-Iweighting)* M + Iweighting * np.eye(M.shape[0])

def get_t_map(i=0, t=10):
    v = np.zeros(M.shape[0]).astype(np.float64)
    v[i] = 1.
    return (v.T @ (np.linalg.matrix_power)).T

def recenter_points(i, pts):
    """recenter them about pt i"""
    pts2 = pts.copy()
    pt = pts[i]
    sign = pts - pt > 0

