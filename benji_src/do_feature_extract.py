import sys
sys.path.append('../nemo-eva/src/')

from feature_extractor import FeatureExtractor
import pandas as pd
import glob

import os
os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FE_FB_copyweights_cube/'


def quick_mixin(my_list, end_per_begin=5):
    """[1,2,3,4,5,6,7,8,9,10], 3 -> [1,2,10,3,4,9,5,6,8,7] roughly"""
    i = 0
    j = len(my_list)-1
    out = []
    end_went = False
    while i < j:
        if i % (end_per_begin-1) == 0 and not end_went:
            out.append(my_list[j])
            j -= 1
            end_went = True
        else:
            out.append(my_list[i])
            i += 1
            end_went = False

    out.append(my_list[i])
    return out

data_dir = '/cluster/scratch/bdayan/GIRG_data/'

results_csv = '/cluster/home/bdayan/girgs/nemo-eva/data-paper/3-cleaned_features/results.csv'
df = pd.read_csv(results_csv)

df = df.loc[df.Model == 'real-world']

socfb_graphs = df.loc[df.Model == 'real-world'].loc[df.Type == 'socfb'].sort_values('Nodes')


graph_name_group_pairs = socfb_graphs[['Graph', 'Type']].to_numpy()
graph_dicts = []
for graph_name, group in graph_name_group_pairs:
    # TODO remove? socfg-nips-ego has avg deg 2.0 which is very small, and for some reason
    #  makes GIRG finding much slower
    #  and idk why but it's no longer in data_dir wtf???
    if graph_name == 'socfb-nips-ego':
        continue
    fn = glob.glob(data_dir + graph_name + '.*')[0]
    print(fn)

    graph_dict = {"Group": group, "FullPath": fn, "Name": graph_name}
    graph_dicts.append(graph_dict)


# TODO put back in?
graph_dicts = quick_mixin(graph_dicts, end_per_begin=7)


# # The data has changed for these two.
# df = df.loc[~df.Graph.isin(['bn-human-BNU_1_0025889_session_2', 'bn-human-BNU_1_0025873_session_1-bg'])]
#
# max_number_of_nodes = 100000
# df = df.loc[df.Nodes < max_number_of_nodes].sort_values('Nodes')
#
#
# # array([['ia-enron-only', 'ia'],
# #        ['bn-macaque-rhesus_brain_1', 'bn'],
# #        ['inf-USAir97', 'inf'],
# # ....
# graph_name_group_pairs = df[['Graph', 'Type']].to_numpy()
# graph_dicts = []
# for graph_name, group in graph_name_group_pairs:
#     fn = glob.glob(data_dir + graph_name + '.*')[0]
#     print(fn)
#     graph_dict = {"Group": group, "FullPath": fn, "Name": graph_name}
#     graph_dicts.append(graph_dict)
#
#
#
# graph_dicts = quick_mixin(graph_dicts, end_per_begin=6)

# graph_name_group_pairs = df.groupby('Graph').Type.unique()
# graph_name_group_pairs = graph_name_group_pairs.apply(lambda x: x[0])
# graph_name_group_pairs = graph_name_group_pairs.reset_index()

# graph_names = sorted(df['Graph'].unique())
# graph_names[:4]

# graph_dicts = []
# for i in range(len(graph_name_group_pairs)):
#     stuff = graph_name_group_pairs.iloc[i]
#     fn = glob.glob(data_dir + stuff.Graph + '.*')[0]
#     print(fn)
#     graph_dict = {"Group": stuff.Type, "FullPath": fn, "Name": stuff.Graph}
#     graph_dicts.append(graph_dict)

# Try first just a few
# graph_dicts = graph_dicts[:3]

import multiprocessing
if __name__ == '__main__':
    print('running on graph dicts:')
    print(graph_dicts)

    fe = FeatureExtractor(graph_dicts)

    print('execute fast write?')
    fe.execute_immediate_write()

    # self = fe
    # if not os.path.exists(self._stagepath):
    #     os.makedirs(self._stagepath)
    #
    # # import multiprocessing.dummy
    # writer_pool = multiprocessing.Pool(1)
    # # writer_pool = multiprocessing.dummy.Pool(1)
    # writer_out = writer_pool.apply_async(self.listener, (self._dict_queue,))
    #
    # graph_dicts = [{'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Caltech36.SpaceOne', 'Name': 'socfb-Caltech36'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Reed98.SpaceOne', 'Name': 'socfb-Reed98'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Haverford76.SpaceOne', 'Name': 'socfb-Haverford76'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Simmons81.SpaceOne', 'Name': 'socfb-Simmons81'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Swarthmore42.SpaceOne', 'Name': 'socfb-Swarthmore42'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Amherst41.SpaceOne', 'Name': 'socfb-Amherst41'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Bowdoin47.SpaceOne', 'Name': 'socfb-Bowdoin47'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Hamilton46.SpaceOne', 'Name': 'socfb-Hamilton46'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Trinity100.SpaceOne', 'Name': 'socfb-Trinity100'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-USFCA72.SpaceOne', 'Name': 'socfb-USFCA72'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Williams40.SpaceOne', 'Name': 'socfb-Williams40'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Oberlin44.SpaceOne', 'Name': 'socfb-Oberlin44'}, {'Group': 'socfb', 'FullPath': '/cluster/scratch/bdayan/GIRG_data/socfb-Wellesley22.SpaceOne', 'Name': 'socfb-Wellesley22'}]
    # for gd in graph_dicts:
    #     fe._execute_one_graph(gd)
    #
    # self._dict_queue.put(None)
    # writer_out.get()
    # writer_pool.close()
    # writer_pool.join()

# sbatch --time=24:00:00 --ntasks=1 --cpus-per-task=12 --mem-per-cpu=2G --wrap="python do_feature_extract.py"
# sbatch --time=24:00:00 --ntasks=1 --cpus-per-task=16 --mem-per-cpu=15G --wrap="python do_feature_extract.py"