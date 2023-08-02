import sys
sys.path.append('../nemo-eva/src/')


import numpy as np

import feature_extractor

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


os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FE_FB_copyweights_cube/'

df = pd.read_csv(os.environ['DATA_PATH'] + '2-features/results.csv')
df.Info = df.Info.apply(lambda temp: {key: eval(var) for key, var in [x.split('=') for x in temp.split('|')]} if (type(temp) is str and '|' in temp) else {
    })
df['alpha'] = df.Info.apply(lambda x: x['alpha'] if 'alpha' in x else 1/float(x['t']) if 't' in x else None)
mini_df = df.loc[df.Model == '1d-copyweight-cube-girg'].sort_values('Nodes').loc[:,['Graph', 'Model', 'Nodes', 'Info'] ]




def _execute_one_graph(graph_name):
    try:
        print(f'processing: {graph_name}', flush=True)
        temp = df.loc[df.Graph == graph_name].sort_values('Model')

        df_path = os.environ['DATA_PATH'] + 'dfs/'
        os.makedirs(df_path, exist_ok=True)


        gd = list(filter(lambda x: x['Name'] == graph_name, do_feature_extract.graph_dicts))[0]
        in_path = gd['FullPath']
        g = nk.readGraph(in_path, nk.Format.EdgeListSpaceOne)
        g = utils.get_largest_component(g)
        nk.overview(g)

        weights = np.array(utils.graph_degrees_to_weights(g))
        for i in range(len(temp)):
            if i == 1:  # only do d=1
                continue
            d = i + 1
            mc_name = graph_name + f'-{d}d'
            df_name = df_path + mc_name + f'_nodes_{g.numberOfNodes()}.csv'
            if os.path.exists(df_name):
                continue


            a, B, pts = utils.get_diffmap_and_points(g, ds=d, process='restrict_uniform_edges')
            pts = points.PointsCube(pts)
            alpha, const = temp.iloc[i].Info['alpha'], temp.iloc[i].Info['const']

            MC = mcmc.MCMC_girg(g, weights.copy(), alpha, const, pts.copy(), pool=False, graph_name=mc_name,
                                failure_prob=0.3, cl_mixin_prob=0.5)

            df_out = MC.ordered_pts_const_alpha_loop(
                # e.g. sqrt(d=3) = 1.73
                # 15 * np.log(2000) = 114
                # 15 * np.log(20000) = 149
                num_pt_proposals=int(15 * np.log(g.numberOfNodes()) * d**0.5),
                num_alpha_proposals=10,
                num_const_proposals=10, num_loops=7+i)


            df_out.to_csv(df_name)

            # save space in pickling :((
            MC.g = None
            MC.A = None
            MC.probs_cl = None
            MC.A_cl = None

            MC.pickle()


    except Exception as e:
        # raise (e)
        print(e)
        pass

def _execute_one_girg_deeper(n, d, d2):
    df_path = os.environ['DATA_PATH'] + 'dfs/'
    os.makedirs(df_path, exist_ok=True)

    mc_name = f'{n} node {d}d cube GIRG gen; {d2}d fit cube GIRG'
    tau = 2.2
    alpha = 1.3
    g, edges, weights, pts, const = generation.generate_GIRG_nk(
        n, d, tau, alpha, weights=None, pts=None, desiredAvgDegree=60,
        points_type=points.PointsCube)

    df_name = df_path + mc_name + '.csv'
    if os.path.exists(df_name):
        return

    print(f'processing: {mc_name}', flush=True)

    a, B, pts = utils.get_diffmap_and_points(g, ds=d2, process='restrict_uniform_edges')
    pts = points.PointsCube(pts)

    MC = mcmc.MCMC_girg(g, weights.copy(), alpha, const, pts.copy(), pool=False, graph_name=mc_name,
                        failure_prob=0.3, cl_mixin_prob=0.5)

    df_out = MC.ordered_pts_const_alpha_loop(
        # e.g. sqrt(d=3) = 1.73
        # 15 * np.log(2000) = 114
        # 15 * np.log(20000) = 149
        num_pt_proposals=int(15 * np.log(g.numberOfNodes()) * d ** 0.5),
        num_alpha_proposals=10,
        num_const_proposals=10, num_loops=7 + d - 1)

    df_out.to_csv(df_name)

    # save space in pickling :((
    MC.g = None
    MC.A = None
    MC.probs_cl = None
    MC.A_cl = None
    MC.pickle()
def _execute_one_girg(n):
    try:
        for d in [1, 2, 3]:
            df_path = os.environ['DATA_PATH'] + 'dfs/'
            os.makedirs(df_path, exist_ok=True)
            for d2 in [1, 2, 3]:

                mc_name = f'{n} node {d}d cube GIRG gen; {d2}d fit cube GIRG'
                tau = 2.2
                alpha=1.3
                g, edges, weights, pts, const = generation.generate_GIRG_nk(
                    n, d, tau, alpha, weights=None, pts=None, desiredAvgDegree=60,
                    points_type=points.PointsCube)

                df_name = df_path + mc_name + '.csv'
                if os.path.exists(df_name):
                    continue
                print(f'processing: {mc_name}', flush=True)

                a, B, pts = utils.get_diffmap_and_points(g, ds=d2, process='restrict_uniform_edges')
                pts = points.PointsCube(pts)

                MC = mcmc.MCMC_girg(g, weights.copy(), alpha, const, pts.copy(), pool=False, graph_name=mc_name,
                                    failure_prob=0.3, cl_mixin_prob=0.5)

                df_out = MC.ordered_pts_const_alpha_loop(
                    # e.g. sqrt(d=3) = 1.73
                    # 15 * np.log(2000) = 114
                    # 15 * np.log(20000) = 149
                    num_pt_proposals=int(15 * np.log(g.numberOfNodes()) * d**0.5),
                    num_alpha_proposals=10,
                    num_const_proposals=10, num_loops=7+d-1)


                df_out.to_csv(df_name)

                # save space in pickling :((
                MC.g = None
                MC.A = None
                MC.probs_cl = None
                MC.A_cl = None
                MC.pickle()

    except Exception as e:
        raise e
        pass

if __name__ == '__main__':
    print('running MCMC')
    os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/MCMC_ordered_girggen/'

    jobs = []
    pool = mp.Pool(processes=12)
    # for j, graph_name in enumerate(mini_df.Graph):
    #     if not j % 2 == 0:
    #         continue
    # for n in [800, 1600, 2500, 4000, 8000, 15000, 25000]:
    for n in [15000, 8000, 25000]:
    # for n in [100, 200]:
        for d in [1, 2, 3]:
            for d2 in [1, 2, 3]:
                job = pool.apply_async(_execute_one_girg_deeper, (n, d, d2))
                jobs.append(job)

    for job in jobs:
        job.get()
