import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph

from benji_girgs.utils import get_perc_lower_common_nhbs

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


def generateEdges(
    weights: np.ndarray,
    positions: Union[Points, np.ndarray],
    alpha: float,
    *,
    const: float = 1.0,
    seed: int = None,
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

def generate_GIRG_nk(n, d, tau, alpha,
                     desiredAvgDegree=None,
                     const=None, weights: Optional[np.ndarray] = None,
                     points_type=PointsTorus,
                     c_implementation=False):
    """Generate a GIRG of n vertices, with power law exponent tau, dimension d and alpha

    NB if the cube version is used, an edge list (pairs (u, v)) is returned instead of an adjacency matrix

    We should try and phase out direct use of cgirg_gen, and use this as a wrapper instead. Code is messy!
    c_implementation only does normal torus GIRGs, no min / mixed / cube GIRGs. Once phased out, all
    the weights (None?) and const/desiredAvgDegree (None?) stuff should only remain in this function.
    """
    # nx.from_numpy_matrix goes from an adjacency matrix. It actually
    # works fine from an upper triangular matrix (with zeros on the diagonal)
    # so all good!
    if points_type is PointsCube:  # This is our only Cube GIRG implementation so far
        gnk, edges, weights, pts, const = generate_GIRG_nk(n, d, tau, alpha, desiredAvgDegree, const, weights, PointsTorus2)
        edges = np.array(upper_triangular_to_edgelist_fastest(edges))
        gnk, edges, weights, pts, const = girg_cube_coupling(gnk, edges, weights, pts, const, alpha, PointsCube)
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
    print(f'const_in: {const_in}')

    pts = generatePositions(n, d)
    pts = points_type(pts)

    # note / np.sqrt(np.sum(weights)), s.t. w_u w_v -> (w_u / sqrt(W)) (w_v / sqrt(W)) = w_u w_v / W
    adj_mat = generateEdges(weights / np.sqrt(np.sum(weights)), pts, alpha, const=const_in)
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

def girg_cube_coupling(gnk, edges: np.ndarray, weights: np.ndarray, pts: Points, const, alpha, cube_type: Type[PointsCube]):
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
