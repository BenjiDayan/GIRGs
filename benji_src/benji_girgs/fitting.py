from itertools import zip_longest
import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph
import networkx as nx

from tqdm import tqdm

from benji_src.benji_girgs.generation import get_dists
from benji_src.benji_girgs import utils
from benji_src.benji_girgs.utils import avg_degree, get_perc_lower_common_nhbs, scale_param
# import os
# if not "NO_CPP_GIRGS" in os.environ:
try:
    from girg_sampling import girgs
except Exception:
    pass




def graph_edge_removed_distances(g, n_samples=1000, use_tqdm=False):
    """
    Randomly samples n_samples edges from the graph, a-b.
    Then removes the edge and computes the shortest path between a and b.
    Returns the unique distance counts of shortest paths between a and b.
    E.g. (infinite distance is replaced by -1)
    unique_dists =  array([ 2,  3,  4, -1])
    counts = array([939,  58,   2,   1])

    For the purposes of std_dev, infinite distances are replaced by the max distance + 2
    """
    edges = list(g.iterEdges())

    distances = []

    for _ in (tqdm(range(n_samples)) if use_tqdm else range(n_samples)):
        edge = edges[np.random.choice(len(edges))]
        a, b = edge
        g.removeEdge(a, b)

        spsp = nk.distance.SPSP(g, [a])
        spsp.run()
        dist = spsp.getDistance(a, b)
        distances.append(dist)
        g.addEdge(a, b)

    # nan_to_num to make sure infinite distances are replaced by large +ve
    distances = np.array(np.nan_to_num(distances)).astype(np.int64)
    unique_dists, dist_counts = np.unique(distances, return_counts=True)
    fixed_infinite_dist = False
    if len(unique_dists) > 1:
        if unique_dists[-1] != unique_dists[-2] + 1:
            unique_dists[-1] = -1

            fixed_infinite_dist = True
    
            distances[distances > unique_dists[-2]] = unique_dists[-1] + 2
            std_distances = np.std(distances)
            # distances[distances > unique_dists[-2]] = -1

    if not fixed_infinite_dist:
        std_distances = np.std(distances)

    return distances, std_distances, unique_dists, dist_counts


def graph_distances(g, n_samples=1000, use_tqdm=False):
    """
    Randomly samples n_samples pairs a, b from the graph.
    Returns the unique distance counts of shortest paths between a and b.
    E.g. (infinite distance is replaced by -1)
    unique_dists =  array([ 2,  3,  4, -1])
    counts = array([939,  58,   2,   1])
    """
    nodes = list(g.iterNodes())
    distances = []

    for _ in (tqdm(range(n_samples)) if use_tqdm else range(n_samples)):
        a, b = np.random.choice(nodes, 2, replace=False)

        spsp = nk.distance.SPSP(g, [a])
        spsp.run()
        dist = spsp.getDistance(a, b)
        distances.append(dist)

    # nan_to_num to make sure infinite distances are replaced by large +ve
    distances = np.array(np.nan_to_num(distances)).astype(np.int64)
    unique_dists, dist_counts = np.unique(distances, return_counts=True)
    fixed_infinite_dist = False
    if len(unique_dists) > 1:
        if unique_dists[-1] != unique_dists[-2] + 1:
            unique_dists[-1] = -1

            fixed_infinite_dist = True
            distances[distances > unique_dists[-2]] = unique_dists[-1] + 2
            # distances[distances > unique_dists[-2]] = -1

    std_distances = np.std(distances)

    return distances, std_distances, unique_dists, dist_counts


def biBFS_sample(g, n_samples=1000, use_tqdm=False):
    """
    Randomly samples n_samples pairs a, b from the graph.
    Returns the unique distance counts of shortest paths between a and b.
    E.g. (infinite distance is replaced by -1)
    unique_dists =  array([ 2,  3,  4, -1])
    counts = array([939,  58,   2,   1])
    """
    g_indices = utils.get_largest_component(g)
    g = utils.quick_subgraph(g, g_indices)

    nodes = list(g.iterNodes())
    sizes = []

    for _ in (tqdm(range(n_samples)) if use_tqdm else range(n_samples)):
        a, b = np.random.choice(nodes, 2, replace=False)
        new_a, new_b, seen_a, seen_b, dist_a, dist_b, met = utils.bi_bfs(g, a, b)
        
        # l_a is the fraction of new_a that intersected with b stuff, e.g. 0.1
        # This means that roughly 10 new a nodes were discovered before one that was
        # also in b, so we count 1/l_a towards the size.
        l_a = len(new_a.intersection(seen_b.union(new_b))) / len(new_a)
        l_b = len(new_b.intersection(seen_a.union(new_a))) / len(new_b)
        sizes.append(len(seen_a) + len(seen_b) + 1/l_a + 1/l_b)

    # unique_sizes, sizes_counts = np.unique(sizes, return_counts=True)

    return sizes

    

    # nan_to_num to make sure infinite distances are replaced by large +ve
    distances = np.array(np.nan_to_num(distances)).astype(np.int64)
    unique_dists, dist_counts = np.unique(distances, return_counts=True)
    fixed_infinite_dist = False
    if len(unique_dists) > 1:
        if unique_dists[-1] != unique_dists[-2] + 1:
            unique_dists[-1] = -1

            fixed_infinite_dist = True
            distances[distances > unique_dists[-2]] = unique_dists[-1] + 2
            std_distances = np.std(distances)
            # distances[distances > unique_dists[-2]] = -1

    if not fixed_infinite_dist:
        std_distances = np.std(distances)

    std_distances = np.std(distances)
    return distances, std_distances, unique_dists, dist_counts


class GirgFitter:
    def gen_new_girg(self):
        new_girg = None
        return new_girg

    def step(self):
        girg = self.gen_new_girg()
        # do some decisions
        ...
        self.t += 1

    def step_n(self, n):
        for i in range(n):
            self.step()

    def fit(self):
        pass


class GirgConstFitter(GirgFitter):
    def __init__(self, target_avg_degree, girg_gen_func):
        self.t = 0
        self.scale = 0.3
        self.target_avg_degree = target_avg_degree
        self.girg_gen_func = girg_gen_func
        self.overshoot = True
        self.verbose=False
        self.const = 1.0

        girg = self.girg_gen_func(self.const)
        self.mu = avg_degree(girg)
        # e.g. 0.2 if mu too big
        self.larger = self.mu < self.target_avg_degree

    def step(self):
        self.const = scale_param(self.const, self.scale, 0.0, self.larger)

        girg = self.girg_gen_func(self.const)
        mu = avg_degree(girg)
        # e.g. 0.2 if mu too big
        larger = mu < self.target_avg_degree

        # girg2 = self.girg_gen_func(self.const)
        # mu2 = avg_degree(girg2)
        # larger2 = mu2 < self.target_avg_degree
        # self.mu = mu2
        self.overshoot = (larger != self.larger)
        self.scale *= (0.7 if self.overshoot else (1/0.7))
        self.scale = min(self.scale, 0.9)

        if self.verbose:
            print(f'mu:{self.mu:.2f} -> mu2:{mu:.2f}; overshoot: {self.overshoot}, scale:{self.scale:.3f}, const: {self.const:.3f}')

        self.larger = larger
        self.mu = mu


    def fit(self):
        """Steps until the average degree is within 3% of the target average degree"""
        while abs(self.mu - self.target_avg_degree) / self.target_avg_degree > 0.03:
            self.step()




class AirportGirgFitter(GirgFitter):
    n = 10317
    d = 2
    tau = 2.103
    num_edges = 6000
    def __init__(self, weights, dists, g_true):
        self.t = 0
        self.temp_scalers = np.array([0.1, 0.1])
        self.scalers = np.array([0.3, 0.3])
        self.alpha = 1.3
        self.const = 1/600

        self.weights = weights # n vector of weights
        self.dists = dists  # nxn matrix of distances
        self.g_true = g_true

        percs = get_perc_lower_common_nhbs(g_true, self.num_edges)
        self.percs_true_median = np.median(percs)
        self.avg_degree_true = avg_degree(g_true)

        print(f'percs_true_median: {self.percs_true_median}, avg_degree_true: {self.avg_degree_true}')

        outer = np.outer(self.weights, self.weights)
        self.p_uv = np.divide(outer, self.dists**d)

    def gen_new_girg(self):
        p_uv = self.const * np.power(self.p_uv, self.alpha)
        p_uv = np.minimum(p_uv, 1)
        unif_mat = np.random.uniform(size=p_uv.shape)
        edges = np.triu((unif_mat < p_uv).astype(np.uint), 1)
        g_girg = nk.nxadapter.nx2nk(nx.from_numpy_array(edges))

        return g_girg

    def step(self):
        g_girg = self.gen_new_girg()
        # do some decisions
        percs = get_perc_lower_common_nhbs(g_girg, self.num_edges)
        percs_median = np.median(percs)
        girg_avg_degree = avg_degree(g_girg)
        print(f'percs_median: {percs_median:.3f}, girg_avg_degree: {girg_avg_degree:.3f}')

        scaly_thing = self.scalers * np.exp(-self.temp_scalers * self.t)
        print(f'scaly_thing: {scaly_thing}')

        # TODO maybe how far not just larger or smaller. Gradient descent esque.
        # 
        larger = percs_median < self.percs_true_median
        print(f'alpha: {"+" if larger else "-"} ', end='')
        self.alpha = scale_param(self.alpha, scaly_thing[0], 1.0, larger)
        larger = self.avg_degree_true > girg_avg_degree
        print(f'c: {"+" if larger else "-"} ')
        self.const = scale_param(self.const, scaly_thing[1], 0, larger)

        print(f'alpha: {self.alpha:.3f}, const: {self.const:.4e}')
        self.t += 1


class AirportLikelihoodGirgFitter(GirgFitter):
    d = 2
    # tau = 2.103
    num_edges = 6000
    alpha_step_size=1e-6
    c_step_size=1e-5
    def __init__(self, weights, dists, g_true):
        self.t = 0
        self.temp_scalers = np.array([0.1, 0.1])
        self.scalers = np.array([0.3, 0.3])
        self.alpha = 1.3
        self.const = 1/600

        self.weights = weights # n vector of weights
        self.dists = dists  # nxn matrix of distances
        self.g_true = g_true
        self.A_true = nx.adjacency_matrix(nk.nxadapter.nk2nx(g_true)).toarray()

        percs = get_perc_lower_common_nhbs(g_true, self.num_edges)
        self.percs_true_median = np.median(percs)
        self.avg_degree_true = avg_degree(g_true)

        print(f'percs_true_median: {self.percs_true_median}, avg_degree_true: {self.avg_degree_true}')

        outer = np.outer(self.weights, self.weights)
        # in case some dists are 0 and outer is 0 - set p_uv_initial to 0
        # if outer isn't 0 we will instead get a massive number, but 
        # that would have happened for very small dists anyway.
        self.p_uv_initial = np.nan_to_num(outer / self.dists**self.d)
        self.p_uv_initial = np.clip(self.p_uv_initial, 1e-10, 1 - 1e-10)
        # np.fill_diagonal(self.p_uv_initial, 0)

    def gen_new_girg(self):
        self.p_uv_pow_alpha = np.power(self.p_uv_initial, self.alpha)
        self.p_uv_no_min = self.const * self.p_uv_pow_alpha
        self.p_uv = np.minimum(self.p_uv_no_min, 1)
        unif_mat = np.random.uniform(size=self.p_uv.shape)
        edges = np.triu((unif_mat < self.p_uv).astype(np.uint), 1)
        g_girg = nk.nxadapter.nx2nk(nx.from_numpy_array(edges))

        return g_girg


    def log_likelihood(self):
        A = self.A_true
        A_bar = -self.A_true + 1
        # no self loops
        np.fill_diagonal(A, 0)
        np.fill_diagonal(A_bar, 0)

        # this is all very painful, maybe we can just clip p_uv not to be 0 or 1
        eps = 1e-7
        p_uv = np.clip(self.p_uv, eps, 1-eps)

        # self.p_uv has some 0s and 1s, so np.log gives some -inf. We
        # do nan_to_num which makes -inf -> large negative number, s.t.
        # A_ij * ?_ij is 0 and not nan when previously ?_ij was -inf.

        # of course if A_ij is not 0, i.e. we have an edge, yet still
        # self.p_uv is 0, we -ve large, and log_likelihood will be
        # -inf probably. :(
        log_likelihood = np.sum(
            A * np.nan_to_num((np.log(p_uv))) +
            A_bar * np.nan_to_num((np.log(1 -p_uv)))
            )

        p_uv_less_than_one = self.p_uv < 1
        dp_uv_dc = self.p_uv_pow_alpha * p_uv_less_than_one
        dp_uv_dalpha = p_uv_less_than_one * p_uv * np.log(self.p_uv_initial)

        # gradient_c = np.sum(np.nan_to_num(A / p_uv) * dp_uv_dc + np.nan_to_num(A_bar / (1 - p_uv)) * (- dp_uv_dc))
        gradient_c = np.sum(
            A / self.const * p_uv_less_than_one +
            A_bar / (1 - p_uv) * (-p_uv / self.const) * p_uv_less_than_one
        )
        gradient_alpha = np.sum(
            A / p_uv * dp_uv_dalpha +
            A_bar / (1 - p_uv) * (- dp_uv_dalpha)
        )

        return log_likelihood, gradient_c, gradient_alpha

    def step(self):
        g_girg = self.gen_new_girg()
        # do some decisions
        log_likelihood, gradient_c, gradient_alpha = self.log_likelihood()
        print(f'log_likelihood: {log_likelihood:.4e}, gradient_c: {gradient_c:.4e}, gradient_alpha: {gradient_alpha:.4e}')
        print(f'g_girg avg deg: {avg_degree(g_girg):.2f}, g_true avg deg {self.avg_degree_true:.2f}')


        scaly_thing = self.scalers * np.exp(-self.temp_scalers * self.t)
        # print(f'scaly_thing: {scaly_thing}')

        larger = gradient_alpha > 0
        print(f'alpha: {"+" if larger else "-"} ', end='')
        self.alpha = scale_param(self.alpha, scaly_thing[0], 1.0, larger)
        # self.alpha += gradient_alpha * self.alpha_step_size
        larger = gradient_c > 0
        # self.const += gradient_c * self.c_step_size
        print(f'c: {"+" if larger else "-"} ')
        self.const = scale_param(self.const, scaly_thing[1], 0, larger)

        print(f'alpha: {self.alpha:.3f}, const: {self.const:.4e}')
        self.t += 1


