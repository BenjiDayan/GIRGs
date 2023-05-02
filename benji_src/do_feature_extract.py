import sys
sys.path.append('../nemo-eva/src/')

from feature_extractor import FeatureExtractor
import pandas as pd
import glob

import os
os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FeatureExtractionOutLCCMini3/'


data_dir = '/cluster/scratch/bdayan/GIRG_data/'

results_csv = '/cluster/home/bdayan/girgs/nemo-eva/data-paper/3-cleaned_features/results.csv'
df = pd.read_csv(results_csv)

df = df.loc[df.Model == 'real-world']

max_number_of_nodes = 100000
df = df.loc[df.Nodes < max_number_of_nodes].sort_values('Nodes')


# array([['ia-enron-only', 'ia'],
#        ['bn-macaque-rhesus_brain_1', 'bn'],
#        ['inf-USAir97', 'inf'],
# ....
graph_name_group_pairs = df[['Graph', 'Type']].to_numpy()
graph_dicts = []
for graph_name, group in graph_name_group_pairs:
    fn = glob.glob(data_dir + graph_name + '.*')[0]
    print(fn)
    graph_dict = {"Group": group, "FullPath": fn, "Name": graph_name}
    graph_dicts.append(graph_dict)

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


graph_dicts = quick_mixin(graph_dicts, end_per_begin=6)

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
if __name__ == '__main__':
    print('running on graph dicts:')
    print(graph_dicts)

    fe = FeatureExtractor(graph_dicts)

    print('execute fast write?')
    fe.execute_immediate_write()

# sbatch --time=24:00:00 --ntasks=1 --cpus-per-task=20 --mem-per-cpu=2G --wrap="python do_feature_extract.py"
# sbatch --time=1-20 --ntasks=1 --cpus-per-task=12 --mem-per-cpu=5000 --wrap="python do_feature_extract.py"