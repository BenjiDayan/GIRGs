import sys
sys.path.append('/cluster/home/bdayan/girgs/nemo-eva/src/')
sys.path+= ['/cluster/apps/nss/gcc-6.3.0/python_gpu/3.8.5', '/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/lib64/python38.zip', '/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/lib64/python3.8', '/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/lib64/python3.8/lib-dynload']
import os
os.chdir('/cluster/home/bdayan/girgs/benji_src/')

import numpy as np

# import feature_extractor


#
# from feature_extractor import FeatureExtractor

import pandas as pd
import glob

import os
import do_feature_extract
import networkit

from benji_girgs import utils, mcmc, generation, points
import networkit as nk
import networkx as nx

import matplotlib.pyplot as plt
import multiprocessing as mp
import torch

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"



import os
os.getcwd()
os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FE_FB_chunglu_with_tau/'

df = pd.read_csv(os.environ['DATA_PATH'] + '3-cleaned_features/results.csv')

df.Info = df.Info.apply(lambda temp: {key: eval(var) for key, var in [x.split('=') for x in temp.split('|')]} if (type(temp) is str and '|' in temp) else {
    })
df['alpha'] = df.Info.apply(lambda x: x['alpha'] if 'alpha' in x else 1/float(x['t']) if 't' in x else None)

df.loc[df.Model == 'real-world'].sort_values('Nodes').loc[:, ['Graph', 'Model', 'Nodes', 'Info']]


names = ['socfb-Reed98', 'socfb-Simmons81', 'socfb-Bowdoin47', 'socfb-Vassar85', 'socfb-Rochester38', 'socfb-Princeton12', 'socfb-Yale4', 'socfb-Columbia2']
# names = ['socfb-Caltech36', 'socfb-Reed98']

# names = ['socfb-Reed98', 'socfb-Simmons81', 'socfb-Swarthmore42',  'socfb-Bowdoin47', 'socfb-Hamilton46', 'socfb-Vassar85', 'socfb-Pepperdine86',  'socfb-Rochester38', 'socfb-Lehigh96', 'socfb-Princeton12', 'socfb-CMU',  'socfb-Yale4', 'socfb-Brown11', 'socfb-Stanford3',  'socfb-Columbia2', 'socfb-Harvard1']

ns = df.loc[df.Model == 'real-world'].sort_values('Nodes').loc[df.Graph.isin(names)].Nodes.values

def extract_graph(name):
    gd = list(filter(lambda x: x['Name'] == name, do_feature_extract.graph_dicts))[0]
    in_path = gd['FullPath']

    g = networkit.readGraph(in_path, networkit.Format.EdgeListSpaceOne)
    # g = utils.get_largest_component(g)

    return g


def fit_to_graph(g, d, mc_name):
    try:
        alpha=1.3
        const=0.1

        a, B, pts = utils.get_diffmap_and_points(g, ds=d, process='restrict_uniform_edges')
        pts = points.PointsCube(pts)
        weights = np.array(utils.graph_degrees_to_weights(g))

        MC = mcmc.MCMC_girg(g, weights.copy(), alpha, const, pts.copy(), pool=False, graph_name=mc_name,
                            failure_prob=0.3, cl_mixin_prob=0.5)

        MC.to_pytorch()

        lr = 1e-4
        if d == 1:
            lr = 3e-6
        if d == 2:
            lr = 1e-4
        if d == 3:
            lr = 1e-4

        df = MC.ordered_pts_const_alpha_loop_pytorch(lr=lr, use_tqdm=False, num_loops=20, num_alpha_proposals=5)
        df.to_csv(os.environ['DATA_PATH'] + 'dfs/' + mc_name + '.csv')

        # save space in pickling :((
        MC.g = None
        MC.A = None
        MC.probs_cl = None
        MC.A_cl = None
        MC.pickle()

    except Exception as e:
        print(f'ERRRRRRROR: {e}')



if __name__ == '__main__':
    print('running experiment')
    os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/MCMC_ordered_scientific6_just_real/'

    if not os.path.exists(os.environ['DATA_PATH']):
        os.makedirs(os.environ['DATA_PATH'])

    if not os.path.exists(os.environ['DATA_PATH'] + 'dfs/'):
        os.makedirs(os.environ['DATA_PATH'] + 'dfs/')

    jobs = []
    pool = mp.Pool(processes=13)

    for name in names:
        print(name)
        g = extract_graph(name)
        g = utils.get_largest_component(g)

        # g_cl = generation.fit_chung_lu(g)
        # g_cl = utils.get_largest_component(g_cl)

        weights = np.array(utils.graph_degrees_to_weights(g))
        n = g.numberOfNodes()

        # g_GIRGs = []
        # for d in [1,2,3]:
        #     tau = 2.5  # ignored
        #     alpha = 1.3
        #     g_GIRG, _, _, _, _ = generation.generate_GIRG_nk(
        #         n, d, tau, alpha, weights=weights, pts=None, desiredAvgDegree=60,
        #         points_type=points.PointsCube)
        #     g_GIRG = utils.get_largest_component(g_GIRG)
        #     g_GIRGs.append((g_GIRG, d))


        for d in [1,2,3]:
            n = g.numberOfNodes()
            mc_name = f'{name}_nodes_{n}_{d}d_cube_GIRG_fit'
            jobs.append(pool.apply_async(fit_to_graph, args=(g, d, mc_name)))

            # n = g_cl.numberOfNodes()
            # mc_name = f'{name}_CL_nodes_{n}_{d}d_cube_GIRG_fit'
            # jobs.append(pool.apply_async(fit_to_graph, args=(g_cl, d, mc_name)))
            #
            # for g_GIRG, d_gen in g_GIRGs:
            #     n = g_GIRG.numberOfNodes()
            #     mc_name = f'{name}_{d_gen}d_cube_GIRG_nodes_{n}_{d}d_cube_GIRG_fit'
            #     jobs.append(pool.apply_async(fit_to_graph, args=(g_GIRG, d, mc_name)))


    for job in jobs:
        job.get()

# sbatch --time=24:00:00 --ntasks=1 --cpus-per-task=13 --mem-per-cpu=8G --wrap="python do_PEC_scientific_comparison.py"