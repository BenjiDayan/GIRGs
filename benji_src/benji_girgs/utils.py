from itertools import zip_longest
import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph
import networkx as nx

from benji_src.benji_girgs.generation import get_dists
# import os
# if not "NO_CPP_GIRGS" in os.environ:
try:
    from girg_sampling import girgs
except Exception:
    pass


def avg_degree(g: Graph):
    return np.mean(nk.centrality.DegreeCentrality(g).run().scores())


# sample_edge_stuff but with knowledge of the graph?
def sample_edge_stuff_complex(g, num_edges, weights, edges, pts):
    cni = nk.linkprediction.CommonNeighborsIndex(g)

    degrees = np.array([g.degree(node) for node in g.iterNodes()]).astype(np.int64)
    stuff = []
    lambdas = np.arange(0.1, 1.0, 0.1)
    for i in np.random.choice(len(edges), num_edges):
        a, b = edges[i]
        degs = [g.degree(a), g.degree(b)]
        if degs[0] > degs[1]:
            a, b = b, a
        dist = get_dists(np.stack([pts[a], pts[b]]), n**(1/d))[0, 1]

        a_nhbs = np.array(list(g.iterNeighbors(a)))
        b_nhbs = np.array(list(g.iterNeighbors(b)))
        a_weights = degrees[a_nhbs]
        lambda_intersects = []
        for l in lambdas:
            a_big_nhbs = a_nhbs[a_weights > degrees[a] * l]
            lambda_intersects.append(
                (l, 
                 len(set(b_nhbs).intersection(set(a_big_nhbs)))/len(a_big_nhbs)
                )
            )
        stuff.append((min(degs), max(degs), int(cni.run(a, b)), dist, lambda_intersects))
        

    return stuff

def sample_edge_stuff(g: Graph, num_edges: int):
    """Samples num_edges randomly from g, and finds number of common neighbors
    b/w the two (and their degrees)
    """
    cni = nk.linkprediction.CommonNeighborsIndex(g)

    edges = list(g.iterEdges())
    degrees = np.array([g.degree(node) for node in g.iterNodes()]).astype(np.int64)
    stuff = []
    lambdas = np.arange(0.1, 1.0, 0.1)
    for i in np.random.choice(len(edges), num_edges):
        a, b = edges[i]
        degs = [g.degree(a), g.degree(b)]
        if degs[0] > degs[1]:
            a, b = b, a

        a_nhbs = np.array(list(g.iterNeighbors(a)))
        b_nhbs = np.array(list(g.iterNeighbors(b)))
        a_weights = degrees[a_nhbs]
        lambda_intersects = []
        for l in lambdas:
            a_big_nhbs = a_nhbs[a_weights > degrees[a] * l]
            lambda_intersects.append(
                (l, 
                 len(set(b_nhbs).intersection(set(a_big_nhbs)))/len(a_big_nhbs)
                )
            )
        stuff.append((min(degs), max(degs), int(cni.run(a, b)), lambda_intersects))
        

    return stuff

def get_perc_lower_common_nhbs(g: Graph, num_edges):
    """# of common nhbs between random edges a-b, divided by
    number of nhbs of the smaller degree node"""
    stuff = sample_edge_stuff(g, num_edges)
    return [x[2]/x[0] for x in stuff]

# TODO deprecated
def sample_edge_stuff2(g, num_edges):
    stuff = sample_edge_stuff(g, num_edges)
    degrees = np.array([g.degree(node) for node in g.iterNodes()]).astype(np.int64)
    qs = np.quantile([g.degree(node) for node in g.iterNodes()], [0.15, 0.5, 0.85])
    return [x for x in stuff if x[0] < qs[0] and x[1] > qs[-1] ]

# TODO remove?
def get_common_nb_percs2(g, num_edges):
    stuff = sample_edge_stuff(g, num_edges)
    return [x[2]/x[0] for x in stuff]


#TODO what is this function?
def sample_possible_triangles(g, num_triangles):
    nodes = list(g.iterNodes())
    stuff = []
    for c in np.random.choice(nodes, num_triangles):
        c_deg = g.degree(c)
        if c_deg < 2:
            continue
    
        c_nhbs = list(g.iterNeighbors(c))
        a, b = np.random.choice(len(c_nhbs), 2, replace=False)
        a, b = c_nhbs[a], c_nhbs[b]

        a_deg, b_deg= g.degree(a), g.degree(b)
        stuff.append((a_deg, b_deg, c_deg, g.hasEdge(a, b)))

    return stuff


# This is a simple way to fit alpha
def fit_girg_alpha(g_true: nk.Graph, d, tau, num_edges=6000):
    """Iteratively improves on an initial guess for alpha,
    fitting to a metric (currently median of percent lower
    common nhbs)
    """
    n = g_true.numberOfNodes()
    alpha = 2.0

    dd = sorted(nk.centrality.DegreeCentrality(g_true).run().scores(), reverse=True)
    desired_degree = np.mean(dd)

    weights = girgs.generateWeights(n, tau)
    pts = girgs.generatePositions(n, d)

    percs_true = get_perc_lower_common_nhbs(g_true, num_edges)
    percs_true_median = np.median(percs_true)
    print(f'percs_true_median: {percs_true_median}')

    t = 0
    k = 0.1
    l = 0.3

    for i in range(10):
        print(i)
        scale = girgs.scaleWeights(weights, desired_degree, d, alpha)
        weights = list(np.array(weights) * scale)
        edges = girgs.generateEdges(weights, pts, alpha)
        g = nk.nxadapter.nx2nk(nx.from_edgelist(edges))
        percs = get_perc_lower_common_nhbs(g, num_edges)
        percs_median = np.median(percs)
        print(f'alpha: {alpha:.3f}, percs_median: {percs_median:.3f}')
        
        scaly_thing = l*np.exp(-k * t)
        print(f'scaly_thing: {scaly_thing}')

        if percs_median > percs_true_median:
            
            alpha = 1 + (alpha - 1) * (1 - scaly_thing)
        else:
            alpha = 1 + (alpha - 1) * (1 - scaly_thing)**(-1)
            
        t += 1


# This is a simple way to fit alpha
def fit_girg_general(g_true: nk.Graph, d, tau, num_edges=6000):
    """Iteratively improves on an initial guess for alpha,
    fitting to a metric (currently median of percent lower
    common nhbs)
    """
    n = g_true.numberOfNodes()
    alpha = 2.0

    dd = sorted(nk.centrality.DegreeCentrality(g_true).run().scores(), reverse=True)
    desired_degree = np.mean(dd)

    weights = girgs.generateWeights(n, tau)
    pts = girgs.generatePositions(n, d)

    percs_true = get_perc_lower_common_nhbs(g_true, num_edges)
    percs_true_median = np.median(percs_true)
    print(f'percs_true_median: {percs_true_median}')

    t = 0
    k = 0.1
    l = 0.3

    for i in range(10):
        print(i)
        scale = girgs.scaleWeights(weights, desired_degree, d, alpha)
        weights = list(np.array(weights) * scale)
        edges = girgs.generateEdges(weights, pts, alpha)
        g = nk.nxadapter.nx2nk(nx.from_edgelist(edges))
        percs = get_perc_lower_common_nhbs(g, num_edges)
        percs_median = np.median(percs)
        print(f'alpha: {alpha:.3f}, percs_median: {percs_median:.3f}')
        
        scaly_thing = l*np.exp(-k * t)
        print(f'scaly_thing: {scaly_thing}')

        if percs_median > percs_true_median:     
            alpha = 1 + (alpha - 1) * (1 - scaly_thing)
        else:
            alpha = 1 + (alpha - 1) * (1 - scaly_thing)**(-1)
            
        t += 1

def scale_param(param, scale, base, larger=True):
    if larger:
        return base + (param - base) * (1 - scale)**(-1)
    else:
        return base + (param - base) * (1 - scale)
    


from benji_src.benji_girgs.utils import get_perc_lower_common_nhbs, avg_degree, scale_param

class GirgFitter:
    def __init__(self):
        self.t = 0
        self.temp_scalers = [0.1]
        self.scalers = [0.3]

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




def expected_node_weight_func(n, tau, alpha, d):
    w_mean = (tau-1)/(tau-2)
    # w_alpha_mean_empirical = np.mean(weights**alpha)
    w_alpha_mean = (tau-1)/(tau-1-alpha)

    a = w_mean
    b = w_mean / (alpha - 1)
    # c_emp = w_alpha_mean_empirical  * 2**(d*(alpha-1)) * n**(1-alpha) / (1 - alpha)
    # c = c_emp * w_alpha_mean / w_alpha_mean_empirical
    c = w_alpha_mean * 2**(d*(alpha-1)) * n**(1-alpha) / (1 - alpha)

    def func(weight):
        expected_degree = (2**d) * ((a + b) * weight + c * weight**alpha)
        return expected_degree
    
    return func
    

# interestingly this doesn't depend on tau?
def expected_node_degree_c(n, tau, alpha, d, const):
    w_mean = (tau-1)/(tau-2)
    # w_mean = np.mean(weights)
    # w_alpha_mean_empirical = np.mean(weights**alpha)
    w_alpha_mean = (tau-1)/(tau-1-alpha)


    # W ~= n w_mean => Sum_v n w_v/W ~= 1
    a =  const**(1/alpha)
    b =  const**(1/alpha) / (alpha - 1)
    # c_emp = (n/(np.sum(weights)**(alpha))) *  w_alpha_mean_empirical  * const / (1 - alpha)
    # W^alpha ~= n w_alpha_mean => Sum_v n w_v^alpha W^{-alpha} approxeq n**(1-alpha)
    c = n**(1-alpha) * const / (1-alpha)

    def func(weight):
        expected_degree = (2**d) * ((a + b) * weight + c * weight**alpha)
        return expected_degree
    
    return func

# p(r | w_u, w_v, x_u)
def p_r_given_wu_wv(n, tau, alpha, d, const, w_u, w_v,):
    w_mean = (tau-1)/(tau-2)
    # W ~ n w_mean
    p_uv_given_r = lambda r: \
        np.minimum(1, const * (
               (w_u * w_v / (r**d * n*w_mean))
           )**alpha)


    p_r = lambda r: 2**d * d * r**(d-1)


    a =  const**(1/alpha)
    b =  const**(1/alpha) / (alpha - 1)

    w_mean = (tau-1)/(tau-2)
    E_W = n * w_mean

    w_alpha_mean = (tau-1)/(tau-1-alpha)
    # E[W^alpha] ~= n * E[w^alpha]
    E_W_alpha = n * w_alpha_mean
    p_uv = (2**d) * (
        (a + b) * w_u * w_v / E_W
         + const * (w_u * w_v)**alpha / E_W_alpha
    )

    return lambda r: p_r(r) * p_uv_given_r(r) / p_uv


def weight_thresh_all_pairs_distances(g, weights, weight_thresh):
    indices = np.argwhere(weights < weight_thresh).reshape(-1)

    g2 = quick_subgraph(g, indices)
    apsp = nk.distance.APSP(g2)
    apsp.run()

    dists = np.array(apsp.getDistances()).astype(np.int64)

    return dists

def fractal_dimensions(dists):
    # if there were any infinite distances, they are now -9223372036854775808
    min_dist = dists.min()
    max_dist = dists.max()
    if min_dist < 0:
        dists[dists == min_dist] = max_dist + 10
    n = dists.shape[0]
    cn_r = []
    nchoose2 = n * (n-1)/2
    for r in range(1, max_dist + 1):
        cn_r.append(np.sum(dists < r)/ (2 * nchoose2))

    return cn_r



def sort_out_dist_matrix(dists):
    """dists: a square n x n integer matrix of distances.
    
    return: something like
    array([[  1,   5,  30, 131, 431, 354,  28,   0],
           [  1,   9,  47, 202, 479, 231,  11,   0],
           [  1,   4,  19,  86, 317, 486,  67,   0],
           [  1,   0,   0,   0,   0,   0,   0,   0],
           ...
        ])
    I.e. the longest distance was 7 here.
    """
    # Remove - infinities
    new_dists = []
    for vec in dists:
        # distances will look like -9223372036854775808, 0, 1, 2, ...
        distances, counts = np.unique(vec, return_counts=True)
        new_dists.append(counts[distances >= 0])
        
    return np.array(list(zip_longest(*new_dists, fillvalue=0))).T



def simple_bfs_search(g, weights, weight_thresh):
    """Performs bfs search """
    suitable_starting = np.argwhere(weights < weight_thresh).reshape(-1)
    dist_2s = []
    for starting_node in suitable_starting[:10]:
        dists = simple_bfs(g, weights, weight_thresh, starting_node)
        dists2 = {k: len(v) for k, v in dists.items()}
        dist_2s.append(dists2[2])
    return np.median(dist_2s)


def simple_bfs(g, weights, weight_thresh, starting_node):
    """BFS out from starting_node, only accessing nodes with weight < weight_thresh.
    Nodes with higher weights are ignored.
    
    return: dist = {0: [starting_node], 1: [nodes of dist 1], 2: [nodes of dist 2], ...}"""
    dists = {0: [starting_node]}
    visited = set([starting_node])
    dist = 0
    while True:
#         print(dist)
        trigger = False
        new_dist = []
        for node in dists[dist]:
            for nhb in g.iterNeighbors(node):
                if not nhb in visited:
                    trigger = True
                    visited.add(nhb)
                    if weights[nhb] < weight_thresh:
                        new_dist.append(nhb)         
                        
        dist += 1
        dists[dist] = new_dist
        if not trigger:
            break
            
    return dists


def quick_subgraph(g, indices):
    """Create a subgraph of g of just indices, relabelled with node IDs 0, 1, 2, ...
    instead of whatever they were before. Relabelling is necessary to do a subsequent
    distance computation"""
    # g2 = nk.graphtools.subgraphFromNodes(g, indices)
    indices_set = set(indices)
    edges = [(u, nhb) for u in indices for nhb in g.iterNeighbors(u) if nhb in indices_set]
    old2new_nodes = {}
    i = 0
    for node in indices:
        old2new_nodes[node] = i
        i += 1

    g3 = nk.Graph()
    for node in range(i):
        _ = g3.addNode()

    seen = set()
    for u, v in edges:
        edge = (min(u, v), max(u, v))
        if edge not in seen:
            seen.add(edge)
            _ = g3.addEdge(old2new_nodes[u], old2new_nodes[v])
        
    return g3


def compare_two_graphs(g1, g2):
    """Compare two graphs, g1 and g2, by computing the fraction of edges that are
    in g1 but not in g2, and vice versa. This is a symmetric measure."""
    edges1 = set()
    for u in range(g1.numberOfNodes()):
        for v in g1.iterNeighbors(u):
            edges1.add((min(u, v), max(u, v)))
    edges2 = set()
    for u in range(g2.numberOfNodes()):
        for v in g2.iterNeighbors(u):
            edges2.add((min(u, v), max(u, v)))
    return len(edges1 - edges2) / len(edges1), len(edges2 - edges1) / len(edges2)


def get_largest_component(g):
    """Return the largest connected component of g"""
    components = nk.components.ConnectedComponents(g)
    components.run()
    comp_id_to_size = components.getComponentSizes()
    largest_component = np.argmax(list(comp_id_to_size.values()))
    return components.getComponents()[largest_component]