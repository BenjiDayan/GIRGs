import os
import sys
sys.path.append('../nemo-eva/src/')
from benji_girgs.graph_kernels import *
import pickle
import grakel


if __name__ == '__main__':
    folder = 'cube_similarity_plots_WLK_5_n_iter_15_n_per'
    os.makedirs(folder, exist_ok=True)

    for name in df_mini.loc[df_mini.Model=='1d-copyweight-cube-girg'].sort_values('Nodes').Graph.iloc[:100:3]:
        print(name)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        try:
            data = run_experiment(name=name, n_per=15, cl_mixin_prob=0.0, kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
                                  node_labelling_func=lambda g, v: g.degree(v), plot_type='boxplot')
        except Exception as e:
            print(e)
            continue
        row = df.loc[df.Graph == name].sort_values('Model').iloc[0]
        plt.title(f"{name}: {row.Nodes} Nodes")
        plt.ylabel('1 - RW kernel with original graph')
        plt.xlabel('Graph Generating Model')
        plt.savefig(f'{folder}/{name}.png')
        pickle.dump((fig, ax), open(f'{folder}/{name}.pkl', 'wb'))
        pickle.dump(data, open(f'{folder}/{name}.data.pkl', 'wb'))

# sbatch --time=06:00:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=10G --wrap="python do_graph_kernel_plots.py"