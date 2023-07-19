import os
import sys
sys.path.append('../nemo-eva/src/')
from benji_girgs import graph_kernels, utils, points
import pickle
import grakel
import itertools
import matplotlib.pyplot as plt
import do_feature_extract
import networkit as nk
import seaborn as sns



if __name__ == '__main__':
    folder = 'gentorus_RW_with_betweenness'
    os.makedirs(folder, exist_ok=True)

    edge_labelling_func = lambda g: {e: 1 for e in g.iterEdges()}
    node_labelling_func = lambda g: graph_kernels.graph_to_labels(g, metric='betweenness', num_colors=10)

    n_per = 8
    num_colors = 4
    for d in [1, 2, 3, 4]:
        n = 400
        alpha=5.0
        desiredAvgDegree=20.0
        print(f'd: {d}, n: {n}')
        data, info = graph_kernels.multiple_girg_comparisons(
            d=d, n=n, tau=2.5, alpha=alpha, desiredAvgDegree=desiredAvgDegree,
            # kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
            # kernel=grakel.SubgraphMatching(k=5, n_jobs=5),
            kernel=grakel.RandomWalkLabeled(normalize=True, lamda=1e-5,
                                         kernel_type='geometric',
                                         n_jobs=10),
            n_per=n_per, node_labelling_func=lambda
            g: graph_kernels.graph_to_labels(g, metric='betweenness', num_colors=num_colors),
            plot_type='boxplot',
            points_type=points.PointsTorus2)
        name = f'd={d} n={n} alpha={alpha} num_colors={num_colors} n_per={n_per}'
        plt.savefig(f'{folder}/{name}.png')
        with open(f'{folder}/{name}.pkl', 'wb') as file:
            pickle.dump(data, file)

    num_colors=7
    for d in [1, 2, 3, 4]:
        n = 400
        alpha=5.0
        desiredAvgDegree=20.0
        print(f'd: {d}, n: {n}')
        data, info = graph_kernels.multiple_girg_comparisons(
            d=d, n=n, tau=2.5, alpha=alpha, desiredAvgDegree=desiredAvgDegree,
            # kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
            # kernel=grakel.SubgraphMatching(k=5, n_jobs=5),
            kernel=grakel.RandomWalkLabeled(normalize=True, lamda=1e-5,
                                         kernel_type='geometric',
                                         n_jobs=10),
            n_per=n_per, node_labelling_func=lambda
            g: graph_kernels.graph_to_labels(g, metric='betweenness', num_colors=num_colors),
            plot_type='boxplot',
            points_type=points.PointsTorus2)
        name = f'd={d} n={n} alpha={alpha} num_colors={num_colors} n_per={n_per}'
        plt.savefig(f'{folder}/{name}.png')
        with open(f'{folder}/{name}.pkl', 'wb') as file:
            pickle.dump(data, file)

    for d in [1, 2, 3, 4]:
        n = 800
        alpha=5.0
        desiredAvgDegree=30.0
        print(f'd: {d}, n: {n}')
        data, info = graph_kernels.multiple_girg_comparisons(
            d=d, n=n, tau=2.5, alpha=alpha, desiredAvgDegree=desiredAvgDegree,
            # kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
            # kernel=grakel.SubgraphMatching(k=5, n_jobs=5),
            kernel=grakel.RandomWalkLabeled(normalize=True, lamda=1e-5,
                                         kernel_type='geometric',
                                         n_jobs=10),
            n_per=n_per, node_labelling_func=lambda
            g: graph_kernels.graph_to_labels(g, metric='betweenness', num_colors=num_colors),
            plot_type='boxplot',
            points_type=points.PointsTorus2)
        name = f'd={d} n={n} alpha={alpha} num_colors={num_colors} n_per={n_per}'
        plt.savefig(f'{folder}/{name}.png')
        with open(f'{folder}/{name}.pkl', 'wb') as file:
            pickle.dump(data, file)

    for d in [1, 2, 3, 4]:
        n = 2000
        alpha=5.0
        desiredAvgDegree=40.0
        print(f'd: {d}, n: {n}')
        data, info = graph_kernels.multiple_girg_comparisons(
            d=d, n=n, tau=2.5, alpha=alpha, desiredAvgDegree=desiredAvgDegree,
            # kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
            # kernel=grakel.SubgraphMatching(k=5, n_jobs=5),
            kernel=grakel.RandomWalkLabeled(normalize=True, lamda=1e-5,
                                         kernel_type='geometric',
                                         n_jobs=10),
            n_per=n_per, node_labelling_func=lambda
            g: graph_kernels.graph_to_labels(g, metric='betweenness', num_colors=num_colors),
            plot_type='boxplot',
            points_type=points.PointsTorus2)
        name = f'd={d} n={n} alpha={alpha} num_colors={num_colors} n_per={n_per}'
        plt.savefig(f'{folder}/{name}.png')
        with open(f'{folder}/{name}.pkl', 'wb') as file:
            pickle.dump(data, file)



    # d_mains = [1, 2, 3, 4]
    # ns = [40000, 70000]
    # alphas = [1.2, 5.0]
    # for n, alpha, d_main in itertools.product(ns, alphas, d_mains):
    #     print(d_main, alpha, n)
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    #     try:
    #         data, info = graph_kernels.multiple_girg_comparisons(
    #             d=d_main, n=n, tau=2.5, alpha=alpha, desiredAvgDegree=60.0,
    #             kernel=grakel.WeisfeilerLehman(n_iter=5, normalize=True),
    #             n_per=13,
    #             d_max_girgs=4,
    #             node_labelling_func=lambda g: graph_kernels.graph_to_labels(g, num_colors=None),
    #             plot_type='boxplot')
    #     except Exception as e:
    #         print(e)
    #         continue
    #     title = f"d={d_main} alpha={alpha} n={n} GIRG WL-Kernel with others"
    #     plt.title(title)
    #     plt.ylabel('1 - RW kernel with original graph')
    #     plt.xlabel('Graph Generating Model')
    #     plt.savefig(f'{folder}/{title}.png')
    #     pickle.dump((fig, ax), open(f'{folder}/{title}.pkl', 'wb'))
    #     pickle.dump(data, open(f'{folder}/{title}.data.pkl', 'wb'))

# sbatch --time=10:00:00 --ntasks=1 --cpus-per-task=3 --mem-per-cpu=16G --wrap="python do_graph_kernel_plots4.py > betweenness_rw.out"