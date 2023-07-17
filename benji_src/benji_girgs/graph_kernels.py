import grakel
import warnings
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np

import do_feature_extract
from benji_girgs import utils, generation, points
import pandas as pd

import networkit as nk
import networkx as nx

import warnings
import seaborn as sns

from feature_extractor import FeatureExtractor


import os
os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FE_FB_copyweights_cube/'

df = pd.read_csv(os.environ['DATA_PATH'] + '2-features/results.csv')

df.Info = df.Info.apply(lambda temp: {key: var for key, var in [x.split('=') for x in temp.split('|')]} if (type(temp) is str and '|' in temp) else {
    })
df['alpha'] = df.Info.apply(lambda x: float(x['alpha']) if 'alpha' in x else 1/float(x['t']) if 't' in x else None)

df_mini = df.loc[:, ['Graph', 'Model', 'Type', 'Nodes', 'Edges', 'Info', 'alpha']]



# def temp(d=1):
#     # _, _, _, _, _, MC = mcmc.g_initialised_mcmc(g, alpha=1.3, const=1.0, pts_d=d, diffmap_init=False, graph_name='Reed98', failure_prob=0.3, cl_mixin_prob=0.5)
#     # g_MC = MC_to_g_grakel(MC)
#
#     g1, _, _, _, _ = generation.generate_GIRG_nk(n, d, tau, alpha, desiredAvgDegree=desiredAvgDegree,
#                                                                 points_type=points.PointsCube, failure_rate=0.3,
#                                                                 cl_mixin_prob=0.0)
#     A1 = nx.linalg.adjacency_matrix(nk.nxadapter.nk2nx(g1)).todense()
#     g_gk1 = grakel.Graph(A1)
#
#     return get_rw_kernel(g_gk1)
#
# def temp_fit(d=1):
#     info, g_out = fe.fit_ndgirg_general(d, utils.LCC, cube=False, copy_weights=True, verbose=False)(g)
#     A1 = nx.linalg.adjacency_matrix(nk.nxadapter.nk2nx(g_out)).todense()
#     g_gk1 = grakel.Graph(A1)
#
#     return get_rw_kernel(g_gk1)

# def fit_cube_similarity(g, kernel, d=1, name='sofcb-Reed98', cl_mixin_prob=0.0):
#     row = df.loc[df.Graph == name].sort_values('Model').iloc[d-1]
#     alpha = row.alpha
#     const = float(row.Info['const'])
#
#     weights = np.array(utils.graph_degrees_to_weights(g))
#     n = g.numberOfNodes()
#     tau=None
#     g1, _, _, _, _ = generation.generate_GIRG_nk(n, d, tau, alpha, const=const, weights=weights,
#                                                                 points_type=points.PointsCube)
#
#     if cl_mixin_prob > 0.0:
#         g1 = generation.chung_lu_mixin_graph(g1, weights, cl_mixin_prob)
#
#     # nk.overview(g1)
#     A1 = nx.linalg.adjacency_matrix(nk.nxadapter.nk2nx(g1)).todense()
#     g_gk1 = grakel.Graph(A1)
#
#     return kernel.transform([g_gk1])[0, 0]

def get_fit_cube_girg(g, d=1, name='sofcb-Reed98', cl_mixin_prob=0.0):
    row = df.loc[df.Graph == name].sort_values('Model').iloc[d-1]
    alpha = row.alpha
    const = float(row.Info['const'])

    weights = np.array(utils.graph_degrees_to_weights(g))
    n = g.numberOfNodes()
    tau=None
    g1, _, _, _, _ = generation.generate_GIRG_nk(n, d, tau, alpha, const=const, weights=weights,
                                                                points_type=points.PointsCube)

    if cl_mixin_prob > 0.0:
        g1 = generation.chung_lu_mixin_graph(g1, weights, cl_mixin_prob)

    return g1


def run_experiment_meta(original_graph_func, original_graph_kwargs, fit_girg_func, n_per=4, kernel=None, node_labelling_func=None,
                        plot_type='swarmplot', title=None):
    g = original_graph_func(**original_graph_kwargs)
    g = utils.get_largest_component(g)
    g_gk = g_to_grakel(g, node_labelling_func=node_labelling_func)

    if kernel is None:
        kernel = grakel.kernels.RandomWalk(normalize=True, lamda=1e-6, kernel_type='geometric')
        kernel.fit_transform([g_gk])
    else:
        kernel.fit_transform([g_gk])


    def g_out_to_similarity(g_out):
        g_gk1 = g_to_grakel(g_out, node_labelling_func=node_labelling_func)
        return kernel.transform([g_gk1])[0, 0]


    outs = []
    outs.append([])
    print('cl')
    for i in range(n_per):
        g1 = generation.fit_chung_lu(g, seed=i)
        out = g_out_to_similarity(g1)
        print(out)
        outs[-1].append(out)


    for d1 in [1, 2, 3]:
        print(d1)
        outs.append([])
        for i in range(n_per):
            g1 = fit_girg_func(d1)
            out = g_out_to_similarity(g1)
            print(out)
            outs[-1].append(out)


    data = pd.DataFrame(1 - np.array(outs).T, columns=['ChungLu', '1d GIRG', '2d GIRG', '3d GIRG'])
    if plot_type == 'swarmplot':
        sns.swarmplot(data=data)
    elif plot_type == 'boxplot':
        sns.boxplot(data=data)
    else:
        raise Exception('plot_type not recognized - swarmplot or boxplot?')
    plt.yscale('log')
    plt.ylabel('1 - RW kernel with original graph')
    plt.xlabel('Graph Generating Model')
    # plt.title(f'd={d} GIRG')
    if title is not None:
        plt.title(title)

    return data, original_graph_kwargs



def run_experiment(name='socfb-Reed98', n_per=4, cl_mixin_prob=0.0, kernel=None, node_labelling_func=None,
                   plot_type='swarmplot'):
    gd = list(filter(lambda x: x['Name'] == name, do_feature_extract.graph_dicts))[0]
    in_path = gd['FullPath']

    g = nk.readGraph(in_path, nk.Format.EdgeListSpaceOne)
    g = utils.get_largest_component(g)
    # nk.overview(g)

    g_gk = g_to_grakel(g, node_labelling_func=node_labelling_func)

    if kernel is None:
        kernel = grakel.kernels.RandomWalk(normalize=True, lamda=1e-5, kernel_type='geometric')
    kernel.fit_transform([g_gk])

    outs = []

    outs.append([])
    print('cl')
    for i in range(n_per):
        g1 = generation.fit_chung_lu(g, seed=i)
        g_gk1 = g_to_grakel(g1, node_labelling_func=node_labelling_func)
        out = kernel.transform([g_gk1])[0, 0]
        print(out)
        outs[-1].append(out)

    for d in range(1, 4):
        print(d)
        outs.append([])
        for i in range(n_per):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g1 = get_fit_cube_girg(g, d=d, name=name, cl_mixin_prob=cl_mixin_prob)
                g_gk1 = g_to_grakel(g1, node_labelling_func=node_labelling_func)
                out = kernel.transform([g_gk1])[0, 0]
            outs[-1].append(out)
            print(out)

    # data = pd.DataFrame(1 - np.array(outs).T, columns=['ChungLu', '1d GIRG', '2d GIRG', '3d GIRG'])
    # sns.swarmplot(data=data)
    # plt.yscale('log')

    data = pd.DataFrame(1 - np.array(outs).T, columns=['ChungLu', '1d GIRG', '2d GIRG', '3d GIRG'])
    if plot_type == 'swarmplot':
        sns.swarmplot(data=data)
    elif plot_type == 'boxplot':
        sns.boxplot(data=data)
    else:
        raise Exception('plot_type not recognized - swarmplot or boxplot?')
    plt.yscale('log')

    return data


def g_to_grakel(g_nk, node_labelling_func=None):
    gnx = nk.nxadapter.nk2nx(g_nk)
    A = nx.adjacency_matrix(gnx).todense()
    if node_labelling_func is None:
        g_gk = grakel.Graph(A)
    else:
        labels = node_labelling_func(g_nk)
        # g_gk = grakel.Graph(A, node_labels={v: node_labelling_func(g_nk, v) for v in g_nk.iterNodes()})
        g_gk = grakel.Graph(A, node_labels={v: labels[v] for v in g_nk.iterNodes()})
    return g_gk



def graph_to_labels(g, num_colors=5):
    ddarr = np.array(nk.centrality.DegreeCentrality(g).run().scores())
    if num_colors is None:
        return ddarr
    bin_edges = histedges_equalN(ddarr, num_colors)
    colors = np.digitize(ddarr, bin_edges)
    return colors

def graph_to_labels_random(g, num_colors=5):
    colors = np.random.randint(0, num_colors, g.numberOfNodes())
    return colors


def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def multiple_girg_comparisons(d=1, n=100, tau=2.5, alpha=1.5, desiredAvgDegree=50.0,
                              d_max_girgs=3,
                              points_type=points.PointsTorus2, c_implementation=True,
                              kernel=None, n_per=4, node_labelling_func=None,
                              plot_type='swarmplot'):
    """
    This is a cheapish experiment: we don't fit LCC. We use torus
    so that we can use the c_implementation, and don't need to
    fit degree

    if points_type = points.PointsCube, then we can't use c_implementation, you should set it to False
    Also the degrees might be different between the different dimensions.

    plot_type: swarmplot or boxplot (boxplot is better for large n_per)

    NB grakel.WeisfeilerLehman(n_iter=5, normalize=True) is way speedier than the default random walk kernel.

    """
    info = {'n': n, 'tau': tau, 'alpha': alpha, 'desiredAvgDegree': desiredAvgDegree, 'd': d, 'kernel': kernel,
            'points_type':points_type, 'c_implementation':c_implementation}

    def gen_girg(d):
        g, edges, weights, pts, const = generation.generate_GIRG_nk(
            n, d, tau, alpha, desiredAvgDegree=desiredAvgDegree, points_type=points_type, c_implementation=c_implementation)
        return g


    g = gen_girg(d)

    # g = generation.chung_lu_mixin_graph(g, weights, 0.5)
    g = utils.get_largest_component(g)
    nk.overview(g)
    # n = g.numberOfNodes()

    g_gk = g_to_grakel(g, node_labelling_func=node_labelling_func)


    if kernel is None:
        kernel = grakel.kernels.RandomWalk(normalize=True, lamda=1e-6, kernel_type='geometric')
        kernel.fit_transform([g_gk])
    else:
        kernel.fit_transform([g_gk])


    outs = []
    outs.append([])
    print('cl')
    for i in range(n_per):
        g1 = generation.fit_chung_lu(g, seed=i)
        g_gk1 = g_to_grakel(g1, node_labelling_func=node_labelling_func)
        out = kernel.transform([g_gk1])[0, 0]
        print(out)
        outs[-1].append(out)


    for d1 in range(1, d_max_girgs+1):
        print(d1)
        outs.append([])
        for i in range(n_per):
            g1 = gen_girg(d1)
            g_gk1 = g_to_grakel(g1, node_labelling_func=node_labelling_func)
            out = kernel.transform([g_gk1])[0, 0]
            print(out)
            outs[-1].append(out)


    data = pd.DataFrame(1 - np.array(outs).T,
                        columns=['ChungLu'] + [f'{d}d GIRG' for d in range(1, d_max_girgs+1)])
    if plot_type == 'swarmplot':
        sns.swarmplot(data=data)
    elif plot_type == 'boxplot':
        sns.boxplot(data=data, showmeans=True)
    else:
        raise Exception('plot_type not recognized - swarmplot or boxplot?')
    plt.yscale('log')
    plt.ylabel('1 - RW kernel with original graph')
    plt.xlabel('Graph Generating Model')
    plt.title(f'd={d} GIRG')

    return data, info