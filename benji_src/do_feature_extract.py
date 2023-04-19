import sys
sys.path.append('../nemo-eva/src/')

from feature_extractor import FeatureExtractor
import pandas as pd
import glob

import os
os.environ['DATA_PATH'] = '/cluster/home/bdayan/girgs/FeatureExtractionOut/'


data_dir = '/cluster/scratch/bdayan/GIRG_data/'

results_csv = '/cluster/home/bdayan/girgs/nemo-eva/data-paper/3-cleaned_features/results.csv'
df = pd.read_csv(results_csv)

df = df.loc[df.Model == 'real-w`orld']

df = df.sort_values('Nodes')


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
print('running on graph dicts:')
print(graph_dicts)

fe = FeatureExtractor(graph_dicts)

print('execute fast write?')
fe.execute_immediate_write()