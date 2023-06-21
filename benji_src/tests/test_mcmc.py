import pytest
from benji_girgs import generation, points, utils, fitting, mcmc
import networkit as nk
import numpy as np
import networkx as nx


n = 1500
d = 1
tau = 2.1
alpha = 1.2
desiredAvgDegree = 100.0
def test_diffmap():

    g, edges, weights, pts, const = generation.generate_GIRG_nk(n, d, tau, alpha, desiredAvgDegree=desiredAvgDegree,
                                                                points_type=points.PointsCube)

    gnx = nk.nxadapter.nk2nx(g)
    A = nx.adjacency_matrix(gnx).todense()

    const_in = generation.const_conversion(const, alpha, d, true_volume=True)
    p_uv = generation.get_probs(weights / np.sqrt(weights.sum()), pts, alpha, const_in)


    w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.5, eye_or_ones='eye')

    pts_diffmap = np.array([diff_map(i, 10) for i in range(n)])
    pts_diffmap = points.normalise_points_to_cube(pts_diffmap)
    pts_diffmap = points.PointsCube(pts_diffmap[:, 0:1])

    pts_init = points.PointsCube(np.random.uniform(size=pts_diffmap.shape))

    # weights = np.array(utils.graph_degrees_to_weights(g))
    MC = mcmc.MCMC_girg(g, weights, alpha, const, pts_diffmap.copy())
    MC_init = mcmc.MCMC_girg(g, weights, alpha, const, pts_init.copy())
    MC_cheat = mcmc.MCbbMC_girg(g, weights, alpha, const, pts.copy())

    return g, A, weights, const, pts, pts_diffmap, pts_init, MC, MC_init, MC_cheat

def test_diffmap_on_g(g, alpha, const, pts_d=1):
    g, A, weights, const, pts_diffmap, pts_init, MC, MC_init = mcmc.g_diffmap_initialised_mcmc(g, alpha, const, pts_d=pts_d)
    return g, A, weights, const, pts_diffmap, pts_init, MC, MC_init

def MC_to_g_A(MC):
    tau = 2.1 # Ignored
    self = MC
    g, edges, weights, pts, const = generation.generate_GIRG_nk(
        self.n, self.d, tau, self.alpha, weights=self.weights_original,
        const=self.const,
        pts=self.pts,
        points_type=points.PointsCube)

    gnx = nk.nxadapter.nk2nx(g)
    A = nx.linalg.adjacency_matrix(gnx).todense()
    return g, A


def test_temp():
    g, A, weights, const, pts, pts_diffmap, pts_init, MC, MC_init, MC_cheat = test_diffmap()
    g2, A2, weights2, const2, pts2 = generation.generate_GIRG_nk(n, d, tau, alpha, weights=weights, pts=pts, const=const)


