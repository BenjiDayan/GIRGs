import os
import sys
sys.path.append('../nemo-eva/src/')
from benji_girgs import graph_kernels, utils
import pickle
import grakel
import itertools
import matplotlib.pyplot as plt
import do_feature_extract
import networkit as nk
import seaborn as sns

def do_socfb_synthetic2(name='socfb-Caltech36', d=1, n_per=6):
    gd = list(filter(lambda x: x['Name'] == name, do_feature_extract.graph_dicts))[0]
    in_path = gd['FullPath']

    g = nk.readGraph(in_path, nk.Format.EdgeListSpaceOne)
    g = utils.get_largest_component(g)

    cl_mixin_prob=0.0
    g_synthetic = graph_kernels.get_fit_cube_girg(g, d=d, name=name, cl_mixin_prob=cl_mixin_prob)

    print(g.numberOfNodes())
    print(utils.LCC(g))
    print(g_synthetic.numberOfNodes())
    print(utils.LCC(g_synthetic))

    data = graph_kernels.run_experiment(g=g_synthetic, name=name, n_per=n_per,
        kernel=grakel.kernels.RandomWalk(normalize=True, lamda=1e-5, kernel_type='geometric'),
        node_labelling_func=None, plot_type=None)

    print(data.to_csv())
    plt.figure()
    sns.boxplot(data=data)
    plt.title(f'd={d} copyweight cube of {name}: {g.numberOfNodes()} Nodes')
    plt.xlabel('Graph Generative Model')
    plt.ylabel('1 - WL-kernel')
    return data

if __name__ == '__main__':
    folder = 'gencube_copyweight_similarity_plots_RW_n_per_6_v2'
    name = 'socfb-Brandeis99'
    os.makedirs(folder, exist_ok=True)

    for name in ['socfb-Brandeis99', 'socfb-JohnsHopkins55', 'socfb-MIT']:
        print(name)
        for d in [1, 2, 3]:
            print(d)
            data = do_socfb_synthetic2(name=name, d=d, n_per=6)
            plt.yscale('log')
            plt.savefig(f'{folder}/{name} d={d}.png')
            pickle.dump(data, open(f'{folder}/{name} d={d}.data.pkl', 'wb'))


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

# sbatch --time=03:00:00 --ntasks=1 --cpus-per-task=3 --mem-per-cpu=16G --wrap="python do_graph_kernel_plots3.py > do_graph_kernel_plots3.out"