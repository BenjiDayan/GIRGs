import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph

from benji_girgs.utils import get_perc_lower_common_nhbs

try:
    from girg_sampling import girgs
except Exception:
    pass

from typing import List, Optional, Tuple, Union
import random

import powerlaw
import os

from julia.api import Julia
jl_temp = Julia(compiled_modules=False)
from julia import Main as jl
file_path = os.path.dirname(os.path.abspath(__file__)) + '/benji_jl_dists.jl'
jl.file_path = file_path
jl.eval('include(file_path)')


from scipy.spatial.distance import pdist, squareform


class Points(np.ndarray):
    """
    Base Class for (n, d) array of points.
    """
    def dists(self) -> np.ndarray:
        """
        Returns an (n,n) matrix of distances between points
        """
        pass
class PointsTorus(Points):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self):
        return get_dists_julia(self)
    
class PointsCube(Points):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self):
        return get_dists_cube(self)

########## NB
# For PointsMCD and PointsSimpleMixed, we actually exponentiate the distances
# by 1/d. This is because in get_probs, each distance r_uv is raised to the power
# of d. So we need to undo that here, since V_min(r_uv) := r_uv. (This is actually
# a simplification, it's actually 2r <= 1 - (1 - 2r)^d <= 2dr in the paper, which
# they then simplify to just r (as we do here).)
#
# note that the formula P(A u B u C) = 1 - (1-2r)^3 for d=3 e.g. comes from
# P(A u B u C) = P( (A^c n B^c n C^c)^c ) = 1 - P(A^c n B^c n C^c)
# = 1 - P(A^c)P(B^c)P(C^c) = 1 - (1-2r)^3
#
# However our codebase throughout actualy samples from the n**(1/d) side length torus.
# Hence we'd need to replace 1 - (1-2r)^d => n - (n^(1/d) - 2r)^d ~= n^[(d-1)/d] r I guess?
# This kind of makes sense:
#   previously r^d --(r=n^(1/d))-> n = Vol(Torus)
#   now, n^[(d-1)/d] r --(r=n^(1/d))-> n^[(d-1)/d] n^(1/d) = n
#
# We could also derive this from the formula from one section: (2r) * (n^(1/d))^(d-1).
# The intersection of d of these planes has volume basically (2dr) * (n^(1/d))^(d-1),
# which we then scale down to just r * (n^(1/d))^(d-1) = n^[(d-1)/d] r.
#
# This gives us the volume for a Min Max mix:
# So if we group the d dimensions into a1, a2, ..., ak disjoint groups, then the volume
# Vol(r) = n - Prod_{i=1}^k (n^(|a_i|/d) - (2r)^|a_i|) =
#
# I think we'd best use this full formula throughout for consistency
class PointsMCD(Points):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def dists(self):
        n, d = self.shape
        r = get_dists_mcd(self)
        # return (n**((d-1)/d) * r)**(1/d)
        out = n - (n**(1/d) - 2*r)**d
        return out**(1/d)
def get_points_simple_mixed_class(groups):
    """
    E.g. if groups = [[0], [1,2]]
    We get a Min(diff0, Max(diff1, diff2)) distance for absolue torus distances.

    """
    class PointsSimpleMixed(Points):
        my_groups = groups
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            return obj

        def dists(self):
            n, d = self.shape
            r = get_dists_mixed(self, self.my_groups)
            prod = 1
            for group in self.my_groups:
                prod *= (n**(len(group)/d) - (2*r)**len(group))
            out = n - prod
            return out**(1/d)

    return PointsSimpleMixed


# class PointsSimpleMixed(Points):
#     def __new__(cls, input_array):
#         obj = np.asarray(input_array).view(cls)
#         return obj
#
#     def dists(self):
#         return get_dists_simple_mixed(self)



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


def torus_uniform(d: int = 2, n: int = 1000) -> np.ndarray:
    """uniformly sample from the torus"""
    torus_side_length = n**(1/d)
    return np.random.uniform(high=torus_side_length, size=(n, d))



    
# very slow non vectorised
def get_dists_old(torus_points: np.ndarray, torus_side_length):
    """
    torus_points: (n x d) array of points
    """
    dists = pdist(torus_points, 
              lambda a,b: np.linalg.norm(get_torus_path(a, b, torus_side_length), ord=np.inf))
    return squareform(dists)

# lots of extra memory
def get_dists2(torus_points: np.ndarray, torus_side_length):
    """
    torus_points: (n x d) array of points
    """
    n = len(torus_points)
    lpts = np.tile(torus_points, (n,1))
    rpts = np.repeat(torus_points, n, axis=0)
    diff = np.abs(rpts - lpts)
    torus_diff = np.minimum(diff, torus_side_length - diff)
    dists = np.linalg.norm(torus_diff, ord=np.inf, axis=1)
    return dists.reshape(n, n)

# Minimal extra memory
# This one seems hella faster
def get_dists(torus_points: np.ndarray, torus_side_length):
    diff = np.abs(torus_points[:, None, :] - torus_points[None, :, :])
    torus_diff = np.minimum(diff, torus_side_length - diff)
    dists = np.linalg.norm(torus_diff, ord=np.inf, axis=-1)
    return dists

# Fastest is julia version of Minimal extra memory
def get_dists_julia(torus_points: np.ndarray, n=None, d=None):
    torus_points = torus_points.astype(np.float16)
    # jl.get_dists_novars automatically converts torus_side_length to match
    # same eltype as torus_points
    n = n if n is not None else torus_points.shape[0]
    d = d if d is not None else torus_points.shape[1]
    torus_side_length = n**(1/d)
    return jl.get_dists_novars(torus_points, torus_side_length)


def get_dists_cube(points: np.ndarray):
    return np.linalg.norm(points[:, None, :] - points[None, :, :], ord=np.inf, axis=-1)

def get_dists_mcd(points: np.ndarray):
    # return np.min(np.abs(points[:, None, :] - points[None, :, :]), axis=-1)
    points = points.astype(np.float16)
    n, d = points.shape
    torus_side_length = n**(1/d)
    return jl.get_dists_novars_min(points, torus_side_length)


# def get_dists_mcd2(points: np.ndarray):
#     # return np.min(np.abs(points[:, None, :] - points[None, :, :]), axis=-1)
#     points = points.astype(np.float16)
#     n, d = points.shape
#     torus_side_length = np.float16(n**(1/d))
#
#     diff = np.abs.(pts[:, None, :] - pts[None, :, :])
#     torus_diff = min.(diff, torus_side_length. - diff)
#     dists = minimum(torus_diff, dims=3)
#     return reshape(dists, (n, n))
#
#
#     return jl.get_dists_novars_min(points, torus_side_length)


def get_dists_mixed(points: np.ndarray, groups: List[List[int]]):
    """E.g. groups = [[0], [1, 2], [0, 3], ...] means take
    Min(
        Max(|z_0 - z_0'|),
        Max(|z_1 - z_1'|, |z_2 - z_2'|),
        Max(|z_0 - z_0'|, |z_3 - z_3'|),
        ...
    )"""
    d = points.shape[1]
    out = get_dists_julia(points[:, groups[0]])
    for group in groups[1:]:
        out = np.minimum(out, get_dists_julia(points[:, group], d=d))
    return out

def get_torus_path(a, b, torus_side_length):
    """Returns "b - a" in minimal torus path"""
    dist1, dist2 = b-a, (torus_side_length-np.abs(b-a))*(-np.sign(b-a))
    return np.where(np.abs(dist1) < np.abs(dist2), dist1, dist2)


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
    return torus_uniform(dimension, n)


def get_probs(weights: List[float], pts: Points, alpha=2.0, const=1.0):
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

 
def generate_GIRG_nk(n, d, tau, alpha, const=1.0, points_type=PointsTorus):
    """Generate a GIRG of n vertices, with power law exponent tau, dimension d
    and alpha """
    # nx.from_numpy_matrix goes from an adjacency matrix. It actually
    # works fine from an upper triangular matrix (with zeros on the diagonal)
    # so all good!
    weights = generateWeights(n, tau)
    pts = generatePositions(n, d)
    pts = points_type(pts)
    edges = generateEdges(weights, pts, alpha, const=const)
    # num_edges = edges.sum()
    # edges = edges + edges.T
    # edge_list = upper_triangular_to_edgelist_fastest(edges)

    # write edge_list to a temporary file
    with open("temp.txt", "w") as f:
        for u, v in zip(*np.nonzero(edges)):
            f.write(f"{u} {v}\n")

    ### Remove this attempt to do a METIS file - needing number of edges was bad?
    # write edge_list to a temporary file, one line per node
    # so
    # 1 3 5 10 20\n
    # 5 6\n
    # ...
    # means node 0 has edges to 1, 3, 5, 10, 20; node 1 has edges to 5, 6 etc.
    # with open("temp.txt", "w") as f:
    #     for i in range(n):
    #         f.write(f"{n} {num_edges}\n")
    #         f.write(" ".join(map(str, np.nonzero(edges[i])[0])) + "\n")


    # read it in with networkit
    g = nk.readGraph("temp.txt", nk.Format.EdgeListSpaceZero)
    # g = nk.readGraph("temp.txt", nk.Format.METIS)
    # delete the temporary file
    os.remove("temp.txt")

    # g = nk.nxadapter.nx2nk(nx.from_numpy_array(edges))
    return g, edges, weights, pts




def cgirg_gen(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """Generate a GIRG with C-library
    """
    if weights is None:
        weights = girgs.generateWeights(n, tau)
    scaled_weights = weights

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
    # Make graph from edge list (not adjacency matrix)
    gnx = nx.from_edgelist(edges)
    missing_nodes = set(list(range(n)))
    for node in gnx.nodes:
        missing_nodes.remove(node)
    
    for missing_node in missing_nodes:
        gnx.add_node(missing_node)

    gnk = nk.nxadapter.nx2nk(gnx)
    id2gnk = dict((gnx_id, gnk_id) for (gnx_id, gnk_id) in zip(gnx.nodes(), range(gnx.number_of_nodes())))

    return gnk, edges, weights, pts, const, id2gnk

def cgirg_gen_cube_subsection(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """
    We will use the torus C library to generate a GIRG, but then restrict to a smaller cube.
    If you want a 500 point graph with d=2, you should pass in n=2000
    (and weights of length 2000 if needed).
    """
    gnk, edges, weights, pts, const, id2gnk = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)
    pts = np.array(pts)
    pts_mini_idxs = np.argwhere(pts.max(axis=1) < 0.5).reshape(-1)
    pts_mini_idxs_gnk = [id2gnk[i] for i in pts_mini_idxs]
    gnk_mini = nk.graphtools.subgraphFromNodes(gnk, pts_mini_idxs_gnk, compact=True)
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
    gnk, edges, weights, pts, const, id2gnk = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)
    pts = np.array(pts)
    weights = np.array(weights)
    W = np.sum(weights)

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
            gnk.removeEdge(id2gnk[u], id2gnk[v])

    return gnk

def cgirg_gen_cube_coupling(n, d, tau, alpha, desiredAvgDegree=None, const=None, weights: Optional[List[float]] = None):
    """This version is hopefully more efficient, and uses coupling.
    So G is a torus GIRG, G' is a coupled cube GIRG which has a subset of the edges of G.
    If E_uv == 0 then E'_uv == 0 as well.
    If E_uv == 1, then
        E'_uv =     {1 : P'_uv/ P_uv
                    {0 : 1 - P'_uv/ P_uv
    This is where P'_uv == P_uv mostly if r'_uv == r_uv, and P'_uv < P_uv possibly if r'_uv > r_uv, i.e. distance
    on cube might be longer than on torus
    """
    # P_uv = min[1, c( (w_u w_v/W) / r_uv^d)^alpha]
    # But actually we use the scaled weights, so const c disappears.
    # P_uv = min[1, ( (w_u w_v / W) / r_uv^d)^alpha]
    gnk, edges, weights, pts, const, id2gnk = cgirg_gen(n, d, tau, alpha, desiredAvgDegree, const, weights)

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
        gnk.removeEdge(id2gnk[u], id2gnk[v])

    return gnk


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
