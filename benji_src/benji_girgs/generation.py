import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph 

try:
    from girg_sampling import girgs
except Exception:
    pass

from typing import List, Tuple, Union
import random

import powerlaw

from julia.api import Julia
jl_temp = Julia(compiled_modules=False)
from julia import Main as jl
jl.eval('include("benji_jl_dists.jl")')


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


def torus_uniform(d: int=2, n: int=1000) -> np.ndarray:
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
def get_dists_julia(torus_points: np.ndarray):
    torus_points = torus_points.astype(np.float16)
    # jl.get_dists_novars automatically converts torus_side_length to match
    # same eltype as torus_points
    n, d = torus_points.shape
    torus_side_length = n**(1/d)
    return jl.get_dists_novars(torus_points, torus_side_length)


def get_dists_cube(points: np.ndarray):
    return np.linalg.norm(points[:, None, :] - points[None, :, :], ord=np.inf, axis=-1)



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
    return np.triu((unif_mat < probs).astype(np.uint), 1)
    

 
def generate_GIRG_nk(n, d, tau, alpha, const=1.0, points_type=PointsTorus):
    """Generate a GIRG of n vertices, with power law exponent tau, dimesion d
    and alpha """
    # nx.from_numpy_matrix goes from an adjacency matrix. It actually
    # works fine from an upper triangular matrix (with zeros on the diagonal)
    # so all good!
    weights = generateWeights(n, tau)
    pts = generatePositions(n, d)
    pts = points_type(pts)
    edges = generateEdges(weights, pts, alpha, const=const)
    g = nk.nxadapter.nx2nk(nx.from_numpy_array(edges))
    return g, edges, weights, pts




def cgirg_gen(n, d, tau, alpha, desiredAvgDegree=None):
    """Generate a GIRG with C-library"""
    weights = girgs.generateWeights(n, tau)
    scaled_weights = weights
    const = 1
    if desiredAvgDegree is not None:
        const = girgs.scaleWeights(weights, desiredAvgDegree, d, alpha)
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
    b/w the too (and their degrees)
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




if __name__ == '__main__':
    edges, weights, pts = generate_GIRG(n=6000)