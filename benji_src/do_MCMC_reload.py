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

from benji_girgs import utils, mcmc
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

os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/MCMC_run8/'

mini_df = df.loc[df.Model == '1d-copyweight-cube-girg'].sort_values('Nodes').loc[:,['Graph', 'Model', 'Nodes', 'Info'] ]



import multiprocessing
if __name__ == '__main__':
    print('running MCMC')

    for j, graph_name in enumerate(mini_df.Graph):
        if j < 10:
            continue
        # if j > 4:
        #     break
        print(f'processing: {graph_name}')
        temp = df.loc[df.Graph == graph_name].sort_values('Model')

        gd = list(filter(lambda x: x['Name'] == graph_name, do_feature_extract.graph_dicts))[0]
        in_path = gd['FullPath']
        g = nk.readGraph(in_path, nk.Format.EdgeListSpaceOne)
        g = utils.get_largest_component(g)
        nk.overview(g)

        n = g.numberOfNodes()
##################
        # gnx = nk.nxadapter.nk2nx(g)
        # A = nx.linalg.adjacency_matrix(gnx).todense()
        #
        # g_degs = [g.degree(i) for i in range(g.numberOfNodes())]
        #
        # argsorted = np.argsort(g_degs)[::-1]
        # fe = feature_extractor.FeatureExtractor([])
        #
        # cl = fe.fit_chung_lu(g)
        # gnx = nk.nxadapter.nk2nx(cl)
        # A_cl = nx.linalg.adjacency_matrix(gnx).todense()
        # # A_cl = A_cl[:, argsorted][argsorted, :]
        #
        # out, p1, p2 = mcmc.CM(A[:, argsorted][argsorted, :], A_cl)
        # print(p1, p2)
        # print(out)

######################

        for i in range(len(temp)):
            alpha, const = temp.iloc[i].Info['alpha'], temp.iloc[i].Info['const']


            g, A, weights, const, pts, MC = mcmc.g_initialised_mcmc(g, alpha=alpha, const=const, pts_d=i+1,
                                                                    diffmap_init=True, graph_name=graph_name + f'-{i+1}d')

            g_dm, A_dm = MC.MC_to_g_A()
            print(MC.outs[0])
            print(MC.percent_edges_captureds[0])

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            # e.g. Caltech 762 with d=1 is about 300K iterations
            iterations = int(n * 450 * (i+1))
            MC.run_pool(iterations, pool_size=14, jobs_per_worker=100, plot_every=20000, use_tqdm=False)
            MC.pickle()


# sbatch --time=24:00:00 --ntasks=1 --cpus-per-task=15 --mem-per-cpu=4G --wrap="python do_MCMC.py"