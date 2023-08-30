import pandas as pd
import networkit as nk

from benji_girgs import utils, mcmc, generation, points
import networkx as nx

import pickle
import glob
folder = '/cluster/home/bdayan/girgs/MCMC_ordered_girggen/'
fn_pickles = glob.glob(folder + 'pickles/*')
fn_dfs = glob.glob(folder + 'dfs/*')

# i = 1
# fn = fn_pickles[i]
# print(fn)
# with open(fn, 'rb') as f:
#     MC = pickle.load(f)
# fn = fn_dfs[i]
# print(fn)
# df = pd.read_csv(fn)

import re
df_out = pd.DataFrame(columns=['graph', 'd', 'nodes', 'll', 'CM', 'pec', 'alpha'])

for fn_df in fn_dfs:
    name, d, nodes = re.match(f'.*(socfb-.*)-(\d)d_nodes_(\d*).csv', fn_df).groups()
    d, nodes = int(d), int(nodes)
    df = pd.read_csv(fn_df)
    ll, CM, pec, alpha = df.ll, df.out, df.pec, df.alpha
    df_out = df_out.append({'graph': name, 'd': d, 'nodes': nodes, 'll': ll, 'CM': CM, 'pec': pec, 'alpha': alpha}, ignore_index=True)

df_out['final_pec'] = df_out.pec.apply(lambda x: x.iloc[-1])
df_out['final_ll'] = df_out.ll.apply(lambda x: x.iloc[-1])
df_out['final_alpha'] = df_out.alpha.apply(lambda x: x.iloc[-1])



def fix_fn(fn):
    with open(fn, 'rb') as file:
        MC = pickle.load(file)

    name = MC.graph_name[:-3]
    name
    MC.percent_edges_captureds[-1]
    g = nk.readGraph(f'/cluster/scratch/bdayan/GIRG_data/{name}.SpaceOne', nk.Format.EdgeListSpaceOne)
    g = utils.get_largest_component(g)
    MC.g = g
    gnx = nk.nxadapter.nk2nx(g)
    A = nx.adjacency_matrix(gnx).todense()
    MC.A = A

    MC.failure_prob = 0.0
    MC.cl_mixin_prob = 0.0
    for _ in range(8):
        MC.ll, MC.expected_num_edges = MC.calculate_ll()
        MC.calibrate_const()
        MC.const

    g_out, _, out, pec, pefw = MC.get_CM(A, failure_prob=0.0)
    # print(g_out.numberOfEdges())
    # print(g.numberOfEdges())
    # print(pec)
    return MC, out, pec

import os
fn_pickles = sorted(glob.glob(folder + 'pickles/*'), key=os.path.getsize)

df = pd.DataFrame(columns=['graph', 'd', 'nodes', 'll', 'CM', 'pec', 'alpha'])

# fn = fn_pickles[0]
# print(fn)
# MC, out, pec = fix_fn(fn)
# df = df.append({'graph': MC.graph_name, 'd': MC.d, 'nodes': MC.n, 'll': MC.ll, 'CM': out, 'pec': pec, 'alpha': MC.alpha}, ignore_index=True)
# # print(out)
# print(pec)
# df.to_csv('mcmc_ordered6_pecs.csv')
#

for fn in fn_pickles:
    print(fn)
    MC, out, pec = fix_fn(fn)
    df = df.append({'graph': MC.graph_name, 'd': MC.d, 'nodes': MC.n, 'll': MC.ll, 'CM': out, 'pec': pec, 'alpha': MC.alpha}, ignore_index=True)
    # print(out)
    print(pec)
    df.to_csv('mcmc_ordered_girggen_pecs.csv')