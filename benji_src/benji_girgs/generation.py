import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph

from benji_girgs.utils import get_perc_lower_common_nhbs, graph_degrees_to_weights, graph_degrees_to_weights
from benji_girgs import mcmc

try:
    from girg_sampling import girgs
except Exception:
    pass

from typing import List, Optional, Tuple, Union, Type
import random




from benji_girgs.points import Points, PointsTorus, PointsCube, PointsMCD, PointsTorus2, PointsTrue

def torus_uniform(d: int = 2, n: int = 1000) -> np.ndarray:
    """uniformly sample from the torus"""
    torus_side_length = n**(1/d)
    return np.random.uniform(high=torus_side_length, size=(n, d))


def chung_lu_fit_c(g, weights):
    """p_uv = min(1, c wu wv)
    NB we assume that weights are normalised so actually wu = w'u/sqrt(W)
    so no need to normalise.
    we have to fit c so that sum p_uv is about the degree of g.
    We will do this in a dumb way...
    Basically we will assume that there is no need for min.
    Should hold for sparse graphs

    e.g. weights could be the degree sequence


    """
    num_edges = g.numberOfEdges()  # = (1/2) Sum p_uv
    c = 1.0
    # triggered will ensure that c goes just small enough so that probs now less than 1
    triggered = False
    for _ in range(10):
        probs = np.minimum(c * np.outer(weights, weights), 1)
        E_edges = probs.sum() - np.diag(probs).sum()
        E_edges = E_edges / 2
        c = c * (num_edges / E_edges)

    probs = np.minimum(c * np.outer(weights, weights), 1)

    # c_copy = c
    # E_edges = probs.sum() - np.diag(probs).sum()
    # E_edges = E_edges / 2
    # c = c * (num_edges / E_edges)
    #
    # # If all went well, c < c_copy. Otherwise we're at risk of violating the
    # # minimum thi
    return c, probs

def fit_chung_lu(g, seed=None):
    nk.setSeed(seed=42 if seed is None else seed, useThreadId=False)
    return nk.generators.ChungLuGenerator.fit(g).generate()

def chung_lu_get_stuff(g):
    weights = np.array(graph_degrees_to_weights(g))
    weights /= np.sqrt(weights.sum())
    c, probs_cl = chung_lu_fit_c(g, weights)
    chung_lu_ll = g_probs_to_ll(g, probs_cl)
    er_ll = ER_ll(g)

    g_cl = fit_chung_lu(g)
    g_cl_nx = nk.nxadapter.nk2nx(g_cl)
    A_cl = nx.linalg.adjacency_matrix(g_cl_nx).todense()
    return chung_lu_ll, er_ll, A_cl, probs_cl, c

def ER_ll(g):
    n, E = g.numberOfNodes(), g.numberOfEdges()
    p = E / n*(n-1)/2
    probs = np.ones(shape=(n,n))/2
    return g_probs_to_ll(g, probs)

def g_probs_to_ll(g, probs):
    ll = 0
    for u_index in range(g.numberOfNodes()):
        eps=1e-7
        p_u_to_vs = probs[u_index, :]
        p_u_to_vs = np.clip(p_u_to_vs, eps, 1 - eps)
        u_ll = mcmc.MCMC_girg.p_u_to_vs_to_ll(g, u_index, p_u_to_vs)
        ll += u_ll
    return ll

# def g_probs_ll(g, probs):
#     out = 0
#
#     for p_u_to_vs:




def powerlaw_dist(tau=2.5, x_min=1, n=1000):
    """sample from a tau exponent power law distribution
    pdf: prop to x^-(a+1), i.e. tau = a+1
    mean: ((tau-1) x_min)/(tau - 2) for tau > 2
    x_min: support is [x_min, inf]
    size: number of samples to draw
    """
    a = tau-1
    pareto = (np.random.pareto(a, size=n) + 1) * x_min
    return pareto


def quick_seed(seed: Union[None, int]):
    return seed if seed is not None else random.randint(0, (1 << 31) - 1)


def generateWeights(
    n: int, ple: float, *, seed: int = None
) -> List[float]:
    np.random.seed(quick_seed(seed))
    return powerlaw_dist(tau=ple, n=n)


def generatePositions(
    n: int, dimension: int, *, seed: int = None
) -> np.ndarray:
    np.random.seed(quick_seed(seed))
    return np.random.uniform(size=(n, dimension))


def get_probs(weights: np.ndarray, pts: Points, alpha=2.0, const=1.0):
    """
    Computes min(1, w_u w_v / ||x_u - x_v||_inf^d)^alpha
    as a big n x n square matrix (we only need upper triangular bit tho)
    """
    outer = np.outer(weights, weights)
    n, d = pts.shape
    dists = pts.dists()
    p_uv = np.divide(outer, dists**d)  
    p_uv = np.power(p_uv, alpha)
    p_uv = np.minimum(const * p_uv, 1)
    return p_uv


def get_probs_u(weights: np.ndarray, pts: Points, alpha, const, u_index):
    """
    Computes min(1, w_u w_v / ||x_u - x_v||_inf^d)^alpha for a given u
    """
    n, d = pts.shape
    wuwv = weights[u_index] * weights
    dists = pts[u_index].dist(pts)
    p_uv = np.divide(wuwv, dists**d)
    p_uv = np.power(p_uv, alpha)
    p_uv = np.minimum(const * p_uv, 1)
    return p_uv  # note this is a vector of length n, so includes a self loop at u_index

def get_probs_us(weights: np.ndarray, pts: Points, alpha, const, u_index):
    """
    Computes min(1, w_u w_v / ||x_u - x_v||_inf^d)^alpha for a given u
    """
    n, d = pts.shape
    wuwv = np.outer(weights[u_index], weights)
    dists = pts[u_index].dists(pts)
    p_uv = np.divide(wuwv, dists**d)
    p_uv = np.power(p_uv, alpha)
    p_uv = np.minimum(const * p_uv, 1)
    return p_uv  # note this is a vector of length n, so includes a self loop at u_index

################# NB The below note is now deprecated as I changed Torus points in general to be in a cube of side
# length 1 rather than n^(1/d). This is because the CGIRG code does this and it makes it easier to compare.
########  NB in order to match up with the CGIRGs we need to do something like this:
# i.e. we have to adjust our code to 1: scale pts by wbar**(1/d)
# note get_probs function uses a build in torus side length calculation so that
# needs to be overriden.
#
# additionally need to, for cases of high variance wbar, take it as the mean of the
# weights themselves.
# 
#
# n = 1000
# d = 2
# tau=2.2
# alpha = 3.0

# weights = generateWeights(n, tau)
# pts = generatePositions(n, d)
# # wbar = (tau-1)/(tau-2)
# wbar = np.mean(weights)
# pts = pts * (wbar**(1/d))

# outer = np.outer(weights, weights)
# n, d = pts.shape
# dists = get_dists_julia(pts, (n*wbar)**(1/d))
# p_uv = np.divide(outer, dists**d)  
# p_uv = np.minimum(p_uv, 1)
# p_uv = np.power(p_uv, alpha)

# probs = p_uv
# unif_mat = np.random.uniform(size=probs.shape)
# edges = np.triu((unif_mat < probs).astype(np.uint), 1)

# np.mean(weights)
# (tau-1)/(tau-2)

# g = nk.nxadapter.nx2nk(nx.from_numpy_array(edges))
# nk.overview(g)

def characterise_girg_edges(g, adj_mat, weights, pts, const, alpha, d):
    points_type = type(pts)
    true_volume = PointsTrue in points_type.__mro__
    const_in = const_conversion(const, alpha, d=d, true_volume=true_volume)
    probs = get_probs(weights/np.sqrt(np.sum(weights)), pts, alpha, const_in)
    edge_list = upper_triangular_to_edgelist_fastest(adj_mat)
    out_list = []
    for u, v in edge_list:
        out_list.append((u, v, weights[u], weights[v], probs[u, v]))
    return out_list, probs

def e_statistic(g, q):
    """
    In theory we only consider triples (u, v, y) where u, v are small_d, y is big_d, and both
    u?y, v?y should be 100%

    If a 100% edge exists with probability e <= 1, and all edges are 100% edges.
    Then we're looking at triples (u-v) - ? - y
    Given that u-v and say u-y, then v-y is a 100% edge and exists with probability e

    with prob e^2, both u-y, v-y. with prob 2e(1-e), just one of u-y, v-y.
    with prob (1-e)^2, neither u-y, v-y.
    Hence num_missing is # of cases of 2e(1-e), and num_triangles is # of cases of e^2.

    So then e.g. K = T/M = e^2 / 2e(1-e)
    So then e = 2K / (1+2K)

    In actuality it's more like maybe 0.3-0.6 of edges are 100% edges, and the rest are maybe
    0-80%.

    Maybe simplify: 50% of edges are 100% edges, and 50% are 10% edges.
    Then we have
    1: both u,v -y: 0.5 (e^2) + 0.5 (0.1^2 e^2)
    2: one of u,v -y: 0.5 (2e(1-e)) + 0.5 ( 2 * 0.1 e [0.1(1-e) + 0.9] )

    This gives about 2.2 K / (1.01 + 2.01 K)

    """
    degrees = graph_degrees_to_weights(g)
    small_d, big_d = np.quantile(degrees, [q, 1-q])

    num_missing = 0
    num_triangles = 0

    for u in g.iterNodes():
        if g.degree(u) <= small_d:
            u_small_nhbs = set()
            u_big_nhbs = set()
            for x in g.iterNeighbors(u):
                if g.degree(x) <= small_d:
                    u_small_nhbs.add(x)
                elif g.degree(x) >= big_d:
                    u_big_nhbs.add(x)

            for v in u_small_nhbs:
                for y in u_big_nhbs:
                    if g.hasEdge(v, y):
                        num_triangles += 1
                    else:
                        num_missing += 1


    # triangles are double counted: once for u -> v; u -> y and once for v -> u; v -> y
    # missing are single counted: once for u -> v; u -> y but not for v -> y; v -> y
    #  (if it's v -> y which is the missing triangle link)
    num_triangles /= 2

    return num_missing, num_triangles



def generateEdges(
    weights: np.ndarray,
    positions: Union[Points, np.ndarray],
    alpha: float,
    *,
    const: float = 1.0,
    seed: int = None,
    failure_rate: float = 0.0
) -> np.ndarray:
    """Generates edges for a GIRG with given
    weights: (n,) array of weights
    positions: (n, d) array of positions. np.ndarray defaults to 
    PointsTorus but can be another Points subclaass
    returns: (n, n) array of edges
    """
    np.random.seed(quick_seed(seed))
    # Convert to torus points if not already a dist'able instance
    if not issubclass(type(positions), Points):
        positions = PointsTorus(positions)
    probs = get_probs(weights, positions, alpha, const)
    probs *= (1-failure_rate)
    unif_mat = np.random.uniform(size=probs.shape)
    # upper triangular matrix - lower half and the diagonal zeroed
    # Since we want only one coin flip per (i, j) edge not two different
    # coin flips that could give different outputs
    # return np.triu((unif_mat < probs).astype(np.uint), 1)
    return np.triu((unif_mat < probs), 1)


def upper_triangular_to_edgelist(mat: np.ndarray) -> List[Tuple[int, int]]:
    """Converts an upper triangular matrix to an edgelist"""
    return [(i, j) for i, j in zip(*np.triu_indices_from(mat, 1)) if mat[i, j] > 0]

def upper_triangular_to_edgelist_faster(mat: np.ndarray) -> List[Tuple[int, int]]:
    """Converts an upper triangular matrix to an edgelist, very fast"""
    return np.argwhere(mat > 0).tolist()

def upper_triangular_to_edgelist_fastest(mat: np.ndarray) -> List[Tuple[int, int]]:
    """Converts an upper triangular matrix to an edgelist, fastest possible"""
    return list(zip(*np.nonzero(mat)))


def edge_list_to_nk_graph(edge_list: List[Tuple[int, int]], n) -> Graph:
    g = nk.Graph(n)
    for u, v in edge_list:
        g.addEdge(u, v)

    return g


def quick_expected_degree_func(weights, alpha, c):
    """This is the True Volumetric version of the expected degree function that Blasius numerically optimises over
    in order to get a desired average degree.

    The formula is essentially:
    p_uv|Vol = c ((w_u w_v / W) / Vol)^alpha  (given already all the weights)
    so p_uv = V^ + c (...)^alpha [1/(alpha -1) V^ ^ (1 - alpha) - 1/(alpha -1)]
    = c^(1/alpha) (w_u w_v / W) [1 + 1/(alpha -1)] - c/(alpha - 1) (w_u w_v / W)^alpha

    However we must correct for all places where actually the weights are large enough that p_uv > 1 throughout the
    whole Torus. I.e. then we just set p_uv = 1.0.

    The return value is the expected total number of edges in the graph, i.e. the sum of all degrees, divided by 2.
    """
    cross = np.outer(weights, weights)
    W = np.sum(weights)
    cross = cross/W
    # These indices are for correction
    bad = cross > c**(-1/alpha)
    np.fill_diagonal(cross, 0.0)
    out = (c**(1/alpha)) * cross * (1 + 1/(alpha - 1)) - (1 / (alpha - 1)) * c * (cross ** alpha)
    # correct the bad indices
    out[bad] = 1.0
    return out.sum()/2

def const_conversion(const, alpha, d=None, true_volume=False):
    """

    The const c used by Blasius (returned by scaleWeights) is intended to be used actually as c^alpha

    This is necessary because the c from girgs.scaleWeights is actually used for w_i -> c w_i.
    This makes w_u w_v / W -> c w_u w_v / W, and so the true c' (w_u w_v/W / r^d)^alpha is c' = c^alpha

    What's worse if we're using a true volume formulation, we replace c by c * (2**d).
    """
    out = const
    if true_volume:
        if not type(d) is int and d >=1:
            assert("Must provide a dimension d >= 1")
        out *= (2**d)
    return out**alpha

def chung_lu_mixin_graph(g, weights, cl_mixin_prob):
    g_out = nk.Graph(g.numberOfNodes())
    c, probs_cl = chung_lu_fit_c(g, weights / np.sum(weights))
    cl_override_mat = np.random.uniform(size=probs_cl.shape) < cl_mixin_prob
    # we only need to upper triangular one as doing an instersection
    cl_override_mat = np.triu(cl_override_mat, 1)

    cl_edges_mat = np.random.uniform(size=probs_cl.shape) < probs_cl
    cl_edges_mat = np.triu(cl_edges_mat, 1)

    # Simple mixin strategy:
    # For each pair (u,v), flip coin (cl_mixin_prob)
    # If Heads, go with edge/non edge in original, otherwise edge/non edge in cl graph
    # So if edge in both, then edge in new graph; non-edge in both, then non-edge in new graph
    # if edge in one, non-edge in other, then see coin flip

    for u, v in g.iterEdges():
        u, v = min(u, v), max(u, v)
        # if edge also in cl graph, or flipped to be in original then include
        if not cl_override_mat[u, v] or cl_edges_mat[u, v]:
            g_out.addEdge(u, v)

    for u, v in zip(*np.nonzero(cl_edges_mat)):
        # already added edge if in both; just check coin flip for non-edge in original
        if not g.hasEdge(u, v) and cl_override_mat[u, v]:
            g_out.addEdge(u, v)
    return g_out


def generate_GIRG_nk(n, d, tau, alpha,
                     desiredAvgDegree=None,
                     const=None, weights: Optional[np.ndarray] = None,
                     weights_sum: Optional[float] = None,
                     points_type: Points = PointsTorus,
                     pts: Optional[np.ndarray] = None,
                     c_implementation=False,
                     failure_rate=0.0):
    """Generate a GIRG of n vertices, with power law exponent tau, dimension d and alpha

    NB if the cube version is used, an edge list (pairs (u, v)) is returned instead of an adjacency matrix

    We should try and phase out direct use of cgirg_gen, and use this as a wrapper instead. Code is messy!
    c_implementation only does normal torus GIRGs, no min / mixed / cube GIRGs. Once phased out, all
    the weights (None?) and const/desiredAvgDegree (None?) stuff should only remain in this function.

    NB pts should be an np.ndarray, and points_type is really what matters for the geometry.
        -> Don't pass in pts of type Cube and expecting it to work as a cube!

    If weights_sum is passed in, we're actually subsampling a larger n' > n GIRG which has weight_sum W given.
    This is used in top_k_clique estimation
    """
    # nx.from_numpy_matrix goes from an adjacency matrix. It actually
    # works fine from an upper triangular matrix (with zeros on the diagonal)
    # so all good!
    if points_type is PointsCube or (pts is not None and type(pts) is PointsCube):  # This is our only Cube GIRG implementation so far
        gnk, edges, weights, pts, const = generate_GIRG_nk(n, d, tau, alpha,
            desiredAvgDegree=desiredAvgDegree,
            const=const,
            weights=weights,
            weights_sum=weights_sum,
            points_type=PointsTorus2,
            pts=PointsTorus2(pts) if pts is not None else None,
            failure_rate=failure_rate)
        edges = np.array(upper_triangular_to_edgelist_fastest(edges))
        gnk, edges, weights, pts, const = girg_cube_coupling(gnk, edges, weights, pts, const, alpha, PointsCube, W=np.sum(weights))
        return gnk, edges, weights, pts, const

    if c_implementation:
        gnk, edges, weights, pts, const = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)
        return gnk, edges, weights, pts, const

    if weights is None:
        weights = generateWeights(n, tau)

    if const is not None and desiredAvgDegree is not None:
        raise ValueError("Cannot specify both const and desiredAvgDegree")

    if const is None:
        if desiredAvgDegree is not None:
            const = girgs.scaleWeights(weights, desiredAvgDegree, d, alpha)
        else:
            const = 1.0

    # e.g. PointsTorus2 and PointsMCD are both "True Volume", whereas PointsTorus differs by a factor of 2^d.
    # PointsTorus implementation matces CGIRGs however and hence const, so we must scale based off it.
    true_volume = PointsTrue in points_type.__mro__
    const_in = const_conversion(const, alpha, d=d, true_volume=true_volume)
    # const_in = const * (2 ** d) if PointsTrue in points_type.__mro__ else const
    # This is necessary because the c from girgs.scaleWeights is actually used for w_i -> c w_i.
    # This makes w_u w_v / W -> c w_u w_v / W, and so the true c' (w_u w_v/W / r^d)^alpha is c' = c^alpha
    # const_in = const_in ** alpha
    # print(f'const_in: {const_in}')

    if pts is None:
        pts = generatePositions(n, d)
        pts = points_type(pts)

    # note / np.sqrt(np.sum(weights)), s.t. w_u w_v -> (w_u / sqrt(W)) (w_v / sqrt(W)) = w_u w_v / W
    weight_normaliser = np.sqrt(np.sum(weights)) if weights_sum is None else np.sqrt(weights_sum)
    adj_mat = generateEdges(weights / weight_normaliser, pts, alpha, const=const_in, failure_rate=failure_rate)
    # adj_mat = generateEdges(scaled_weights, pts, alpha, const=1.0)

    # g = nk.nxadapter.nx2nk(nx.from_numpy_array(edges))
    g = edge_list_to_nk_graph(zip(*np.nonzero(adj_mat)), n)
    # TODO we used to return const, now we return const_in. Will this mess anything up in the feature extraction?
    return g, adj_mat, weights, pts, const

def cgirg_gen(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """Generate a GIRG with C-library
    """
    if weights is None:
        weights = girgs.generateWeights(n, tau)

    if const is not None and desiredAvgDegree is not None:
        raise ValueError("Cannot specify both const and desiredAvgDegree")
    
    if const is None:
        if desiredAvgDegree is not None:
            const = girgs.scaleWeights(weights, desiredAvgDegree, d, alpha)
        else:  
            const=1.0

    scaled_weights = list(np.array(weights) * const)
        
    pts = girgs.generatePositions(n, d)
    edges = girgs.generateEdges(scaled_weights, pts, alpha)
    # Make graph from edge list
    g = edge_list_to_nk_graph(edges, n)

    return g, edges, weights, pts, const

def cgirg_gen_cube_subsection(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """
    We will use the torus C library to generate a GIRG, but then restrict to a smaller cube.
    If you want a 500 point graph with d=2, you should pass in n=2000
    (and weights of length 2000 if needed).
    """
    gnk, edges, weights, pts, const = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)
    pts = np.array(pts)
    pts_mini_idxs = np.argwhere(pts.max(axis=1) < 0.5).reshape(-1)
    gnk_mini = nk.graphtools.subgraphFromNodes(gnk, pts_mini_idxs, compact=True)
    return gnk_mini, const

def cgirg_gen_cube_coupling_slow(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """This version is hopefully more efficient, and uses coupling.
    So G is a torus GIRG, G' is a coupled cube GIRG which has a subset of the edges of G.
    If E_uv == 0 then E'_uv == 0 as well.
    If E_uv == 1, then
        E'_uv =     {1 : P'_uv/ P_uv
                    {0 : 1 - P'_uv/ P_uv
    This is where P'_uv == P_uv mostly if r'_uv == r_uv, and P'_uv < P_uv possibly if r'_uv > r_uv, i.e. distance
    on cube might be longer than on torus

    NB this code is much slower than cgirg_gen_cube_coupling, which vectorises
    much of the computation, but it has the same effect. I'm keeping it
    here for reference for the moment.
    """
    # P_uv = min[1, c( (w_u w_v/W) / r_uv^d)^alpha]
    # But actually we use the scaled weights, so const c disappears.
    # P_uv = min[1, ( (w_u w_v / W) / r_uv^d)^alpha]
    gnk, edges, weights, pts, const = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)
    pts = np.array(pts)
    weights = np.array(weights)
    W = np.sum(weights)

    edges_out = []

    for u, v in edges:
        xu, xv = pts[u], pts[v]
        wu, wv = weights[u], weights[v]
        diff = np.abs(xu - xv)
        torus_diff = np.minimum(diff, 1 - diff)
        torus_inf_norm = torus_diff.max()
        cube_inf_norm = diff.max()
        p_uv = min(1, ((wu * wv / W) / torus_inf_norm**d)**alpha)
        p_uv_cube = min(1, ((wu * wv / W) / cube_inf_norm ** d) ** alpha)

        p_edge = p_uv_cube/p_uv
        if np.random.rand() > p_edge:
            gnk.removeEdge(u, v)
        else:
            edges_out.append((u, v))

    return gnk, edges_out, weights, pts, const

# TODO I think this is bugged as const is not taken into account
def cgirg_gen_cube_coupling(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """This version is hopefully more efficient, and uses coupling.
    So G is a torus GIRG, G' is a coupled cube GIRG which has a subset of the edges of G.
    If E_uv == 0 then E'_uv == 0 as well.
    If E_uv == 1, then
        E'_uv =     {1 : P'_uv/ P_uv
                    {0 : 1 - P'_uv/ P_uv
    This is where P'_uv == P_uv mostly if r'_uv == r_uv, and P'_uv < P_uv possibly if r'_uv > r_uv, i.e. distance
    on cube might be longer than on torus

    NB desiredAvgDegree (and const by extension) are both intended for a normal non cube GIRG, so they can
    be given, but the outcome average degree will be different (but same ballpark)
    """
    # P_uv = min[1, c( (w_u w_v/W) / r_uv^d)^alpha]
    # But actually we use the scaled weights, so const c disappears.
    # P_uv = min[1, ( (w_u w_v / W) / r_uv^d)^alpha]
    gnk, edges, weights, pts, const = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)

    edges = np.array(edges)
    pts = np.array(pts)
    weights = np.array(weights)
    W = np.sum(weights)

    u, v = edges[:, 0], edges[:, 1]
    xu, xv = pts[u], pts[v]
    wu, wv = weights[u], weights[v]

    diff = np.abs(xu - xv)
    torus_diff = np.stack([diff, 1 - diff]).min(axis=0)
    torus_inf_norm = torus_diff.max(axis=1)
    cube_inf_norm = diff.max(axis=1)

    puv = np.stack([np.ones(cube_inf_norm.shape), ((wu * wv / W) / torus_inf_norm ** d) ** alpha]).min(axis=0)
    puv_cube = np.stack([np.ones(cube_inf_norm.shape), ((wu * wv / W) / cube_inf_norm ** d) ** alpha]).min(axis=0)

    samples = np.random.uniform(size=cube_inf_norm.shape)
    to_remove = samples > puv_cube / puv

    for u, v in edges[to_remove]:
        gnk.removeEdge(u, v)

    return gnk, edges[~to_remove], weights, pts, const

def girg_cube_coupling(gnk, edges: np.ndarray, weights: np.ndarray, pts: Points, const, alpha, cube_type: Type[PointsCube], W: Optional[float] = None):
    """
    edges should be a [(u, v), ...] 2d array of edges
    Uses pts.dist method vs cube_type(pts).dist method as the coupling differentiation
    """
    # # P_uv = min[1, c( (w_u w_v/W) / r_uv^d)^alpha]
    # # But actually we use the scaled weights, so const c disappears.
    # # P_uv = min[1, ( (w_u w_v / W) / r_uv^d)^alpha]
    # gnk, edges, weights, pts, const = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)

    d = pts.shape[1]
    const_in = const_conversion(const, alpha, d=d, true_volume=True)

    if W is None:
        W = np.sum(weights)
    d = pts.shape[1]

    u, v = edges[:, 0], edges[:, 1]
    xu, xv = pts[u], pts[v]
    original_edge_dists = xu.dist(xv)

    pts = cube_type(pts)
    xu, xv = pts[u], pts[v]
    cube_edge_dists = xu.dist(xv)


    wu, wv = weights[u], weights[v]


    puv = np.stack([np.ones(original_edge_dists.shape), const_in * ((wu * wv / W) / original_edge_dists ** d) ** alpha]).min(axis=0)
    puv_cube = np.stack([np.ones(cube_edge_dists.shape), const_in * ((wu * wv / W) / cube_edge_dists ** d) ** alpha]).min(axis=0)


    samples = np.random.uniform(size=cube_edge_dists.shape)
    to_remove = samples > puv_cube / puv

    for u, v in edges[to_remove]:
        gnk.removeEdge(u, v)

    return gnk, edges[~to_remove], weights, pts, const


cgirg_gen_cube = cgirg_gen_cube_coupling

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