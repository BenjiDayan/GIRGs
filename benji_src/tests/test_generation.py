import pytest
from benji_girgs import generation, points, utils
import networkit as nk
import numpy as np

n = 1500
d = 2
tau = 2.1
alpha = 1.2
desiredAvgDegree = 100.0

def test_cube():
    g, edges, weights, pts, const = generation.generate_GIRG_nk(n, d, tau, alpha, desiredAvgDegree=desiredAvgDegree,
                                                                points_type=points.PointsCube)
    nk.overview(g)

####
if __name__ == '__main__':
    n = 1500
    d = 2
    tau = 2.1
    alpha = 1.2
    desiredAvgDegree = 100.0

    # gnk, edges, weights, pts, const = generation.cgirg_gen(n, d, tau, alpha, desiredAvgDegree=20.0)
    # nk.overview(gnk)

    pp = points.get_points_distorted(np.array([1.0, 20.0]))
    g, edges, weights, pts_torus, const = generation.generate_GIRG_nk(n, d, tau, alpha,
                                                                      desiredAvgDegree=desiredAvgDegree, points_type=pp)
    print(const)
    utils.avg_degree(g)

    w, Phi, Psi, diff_map = utils.get_diffmap(g)

    ys = w[1:13] ** 8
    plt.scatter(list(range(1, len(ys) + 1)), ys)