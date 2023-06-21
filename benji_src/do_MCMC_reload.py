import pickle
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

# def quick_mixin(my_list, end_per_begin=5):
#     """[1,2,3,4,5,6,7,8,9,10], 3 -> [1,2,10,3,4,9,5,6,8,7] roughly"""
#     i = 0
#     j = len(my_list)-1
#     out = []
#     end_went = False
#     while i < j:
#         if i % (end_per_begin-1) == 0 and not end_went:
#             out.append(my_list[j])
#             j -= 1
#             end_went = True
#         else:
#             out.append(my_list[i])
#             i += 1
#             end_went = False
#
#     out.append(my_list[i])
#     return out
#
# data_dir = '/cluster/scratch/bdayan/GIRG_data/'
# results_csv = '/cluster/home/bdayan/girgs/nemo-eva/data-paper/3-cleaned_features/results.csv'
# df = pd.read_csv(results_csv)
# df = df.loc[df.Model == 'real-world']
# socfb_graphs = df.loc[df.Model == 'real-world'].loc[df.Type == 'socfb'].sort_values('Nodes')
#
#
# graph_name_group_pairs = socfb_graphs[['Graph', 'Type']].to_numpy()
# graph_dicts = []
# for graph_name, group in graph_name_group_pairs:
#     # TODO remove? socfg-nips-ego has avg deg 2.0 which is very small, and for some reason
#     #  makes GIRG finding much slower
#     #  and idk why but it's no longer in data_dir wtf???
#     if graph_name == 'socfb-nips-ego':
#         continue
#     fn = glob.glob(data_dir + graph_name + '.*')[0]
#     print(fn)
#
#     graph_dict = {"Group": group, "FullPath": fn, "Name": graph_name}
#     graph_dicts.append(graph_dict)
#
#
# # TODO put back in?
# graph_dicts = quick_mixin(graph_dicts, end_per_begin=7)

############################
os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FE_FB_copyweights_cube/'

df = pd.read_csv(os.environ['DATA_PATH'] + '2-features/results.csv')
df.Info = df.Info.apply(lambda temp: {key: eval(var) for key, var in [x.split('=') for x in temp.split('|')]} if (type(temp) is str and '|' in temp) else {
    })
df['alpha'] = df.Info.apply(lambda x: x['alpha'] if 'alpha' in x else 1/float(x['t']) if 't' in x else None)

os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/MCMC_runs/'

mini_df = df.loc[df.Model == '1d-copyweight-cube-girg'].sort_values('Nodes').loc[:,['Graph', 'Model', 'Nodes', 'Info'] ]

pickle_path = '../MCMC_runs/pickles/'

import multiprocessing
if __name__ == '__main__':
    print('running MCMC')

    for fn in os.listdir(pickle_path)[2:]:
        sep_index = fn[::-1].index('-')
        d = int(fn[-sep_index])
        print(f'pickle file: {fn}')
        print(f'd: {d}')
        try:
            with open(pickle_path + fn, 'rb') as file:
                MC = pickle.load(file)
        except Exception:
            continue

        print(fn)
        g = MC.g

        # g = nk.readGraph(in_path, nk.Format.EdgeListSpaceOne)
        # g = utils.get_largest_component(g)

        nk.overview(g)

        n = g.numberOfNodes()



        # fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        #
        # iterations = int(n * 250 * d)
        # MC.run_pool(iterations, pool_size=14, jobs_per_worker=100, plot_every=20000, use_tqdm=False)
        # MC.pickle()

        ######### We redo things with an e to see if it makes a difference.
        # Note this should make our alpha estimation kind of funky but ah well....
        self = MC
        const = self.const
        const_in = generation.const_conversion(const, self.alpha, d=self.d, true_volume=True)
        e = 0.5


        print('calibrating const')
        for _ in range(7):
            # self.ll, self.expected_num_edges = self.calculate_ll()
            ll = 0
            expected_num_edges = 0
            for u_index in range(self.n):
                eps = 1e-7
                p_u_to_vs = generation.get_probs_u(self.weights, self.pts, self.alpha, const_in, u_index)
                p_u_to_vs = e * p_u_to_vs
                expected_num_edges += p_u_to_vs.sum()
                p_u_to_vs = np.clip(p_u_to_vs, eps, 1 - eps)
                u_ll = self.p_u_to_vs_to_ll(self.g, u_index, p_u_to_vs)
                ll += u_ll

            expected_num_edges = expected_num_edges/2

            print(f'const: {const}, expected_num_edges: {expected_num_edges}')
            ratio = expected_num_edges / self.g.numberOfEdges()
            const = const / ratio
            const_in = generation.const_conversion(const, self.alpha, d=self.d, true_volume=True)

        print(f'final LL: {ll}')

        g_out, A_out, out, percent_edges_captured, percent_fake_edges_wrong = self.get_CM(self.A)
        print(f'MC pec, pfew before: {percent_edges_captured}, {percent_fake_edges_wrong}')
        print(out)

        # g_out, A_out = self.MC_to_g_A()
        tau = 2.1  # Ignored
        g_out, edges, weights, pts, const = generation.generate_GIRG_nk(
            self.n, self.d, tau, self.alpha, weights=self.weights_original,
            const=const,
            pts=self.pts,
            points_type=points.PointsCube,
            e=e)


        gnx = nk.nxadapter.nk2nx(g_out)
        A_out = nx.linalg.adjacency_matrix(gnx).todense()
        out, percent_edges_captured, percent_fake_edges_wrong = mcmc.CM(self.A, A_out)

        print(f'MC pec, pfew after: {percent_edges_captured}, {percent_fake_edges_wrong}')
        print(out)

        print()

        ###################

        # for i in range(len(temp)):
        #     alpha, const = temp.iloc[i].Info['alpha'], temp.iloc[i].Info['const']
        #
        #
        #     g, A, weights, const, pts, MC = mcmc.g_initialised_mcmc(g, alpha=alpha, const=const, pts_d=i+1,
        #                                                             diffmap_init=True, graph_name=graph_name + f'-{i+1}d')
        #
        #     g_dm, A_dm = MC.MC_to_g_A()
        #     print(MC.outs[0])
        #     print(MC.percent_edges_captureds[0])
        #
        #     fig, ax1 = plt.subplots()
        #     ax2 = ax1.twinx()
        #
        #     # e.g. Caltech 762 with d=1 is about 300K iterations
        #     iterations = int(n * 450 * (i+1))
        #     MC.run_pool(iterations, pool_size=14, jobs_per_worker=100, plot_every=20000, use_tqdm=False)
        #     MC.pickle()


# sbatch --time=24:00:00 --ntasks=1 --cpus-per-task=15 --mem-per-cpu=4G --wrap="python do_MCMC_reload.py"