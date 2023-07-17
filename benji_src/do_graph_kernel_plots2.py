import os
import sys
sys.path.append('../nemo-eva/src/')
from benji_girgs import graph_kernels
import pickle
import grakel
import itertools
import matplotlib.pyplot as plt


if __name__ == '__main__':
    folder = 'gentorus_similarity_plots_WLK_n_per_9'
    os.makedirs(folder, exist_ok=True)

    d_mains = [1, 2, 3, 4]
    ns = [3000, 10000, 40000]
    alphas = [1.2, 5.0]
    for n, alpha, d_main in itertools.product(ns, alphas, d_mains):
        print(d_main, alpha, n)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        try:
            data, info = graph_kernels.multiple_girg_comparisons(
                d=d_main, n=n, tau=2.5, alpha=alpha, desiredAvgDegree=60.0,
                kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
                n_per=9,
                d_max_girgs=4,
                node_labelling_func=lambda g: graph_kernels.graph_to_labels(g, num_colors=None),
                plot_type='boxplot')
        except Exception as e:
            print(e)
            continue
        title = f"d={d_main} alpha={alpha} n={n} GIRG WL-Kernel with others"
        plt.title(title)
        plt.ylabel('1 - RW kernel with original graph')
        plt.xlabel('Graph Generating Model')
        plt.savefig(f'{folder}/{title}.png')
        pickle.dump((fig, ax), open(f'{folder}/{title}.pkl', 'wb'))
        pickle.dump(data, open(f'{folder}/{title}.data.pkl', 'wb'))

# sbatch --time=10:00:00 --ntasks=1 --cpus-per-task=2 --mem-per-cpu=16G --wrap="python do_graph_kernel_plots2.py"