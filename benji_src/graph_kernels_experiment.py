import os
import sys
from benji_girgs.graph_kernels import *
import pickle


if __name__ == '__main__':
    folder = 'graph_kernels_experiment'
    os.makedirs(folder, exist_ok=True)

    n=5000
    tau=2.5
    alpha=1.5
    desiredAvgDegree=50.0

    for d in [1, 2, 3]:
        print(d)
        fig, ax = plt.subplots(1, 1)
        try:
            data, info = multiple_girg_comparisons(
                d, n, tau, alpha, desiredAvgDegree, kernel=None,
            points_type = points.PointsCube, c_implementation=False,)
        except Exception as e:
            print(e)
            continue
        name = f'{d}d GIRG n={n} tau={tau} alpha={alpha} desiredAvgDegree={desiredAvgDegree} cube'
        plt.title(name)
        plt.ylabel('1 - RW kernel with original graph')
        plt.xlabel('Graph Generating Model')
        plt.savefig(f'{folder}/{name}.png')
        pickle.dump((fig, ax), open(f'{folder}/{name}.pkl', 'wb'))
        pickle.dump((data, info), open(f'{folder}/{name}.data.pkl', 'wb'))

# sbatch --time=04:00:00 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=16G --wrap="python graph_kernels_experiment.py"