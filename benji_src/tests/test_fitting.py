import pytest
from benji_girgs import generation, points, utils, fitting
import networkit as nk
import numpy as np
import networkx as nx

n = 1500
d = 1
tau = 2.1
alpha = 1.2
desiredAvgDegree = 100.0
def test_simple_cube_mcmc():

    g, edges, weights, pts, const = generation.generate_GIRG_nk(n, d, tau, alpha, desiredAvgDegree=desiredAvgDegree,
                                                                points_type=points.PointsCube)

    gnx = nk.nxadapter.nk2nx(g)
    A = nx.adjacency_matrix(gnx).todense()

    const_in = generation.const_conversion(const, alpha, d, true_volume=True)
    p_uv = generation.get_probs(weights / np.sqrt(weights.sum()), pts, alpha, const_in)


    w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.5, eye_or_ones='eye')

    pts_diffmap = np.array([diff_map(i, 5) for i in range(n)])
    pts_diffmap = points.normalise_points_to_cube(pts_diffmap)
    pts_diffmap = points.PointsTorus2(pts_diffmap[:, 0:1])

    pts_diffmap_final, lls, num_acceptances = fitting.mcmc_girg(A, weights, alpha, d, const, pts_diffmap,
                                                                n_steps=500, ll_every=20)


if __name__ == '__main__':
    # b = 2.0
    # n = 1500
    # d = 1
    # tau = 2.1
    # alpha = 1.2
    # desiredAvgDegree = 100.0
    #
    # g, edges, weights, pts, const = generation.generate_GIRG_nk(n, d, tau, alpha, desiredAvgDegree=desiredAvgDegree,
    #                                                             points_type=points.PointsTorus2)
    # nk.overview(g)
    #
    # gnx = nk.nxadapter.nk2nx(g)
    # A = nx.adjacency_matrix(gnx).todense()
    #
    # const_in = generation.const_conversion(const, alpha, d, true_volume=True)
    # p_uv = generation.get_probs(weights / np.sqrt(weights.sum()), pts, alpha, const_in)
    #
    # print(fitting.girg_loglikelihood(A, p_uv))
    #
    #
    # pts_init = points.PointsTorus2(np.random.uniform(size=(n, d)))
    #
    # pts_init2 = pts_init.copy()
    # pts_final, lls, num_acceptances = fitting.mcmc_girg(A, weights, alpha, d, const, pts_init, n_steps=20)


    test_simple_cube_mcmc()