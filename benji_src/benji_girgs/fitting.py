from itertools import zip_longest
import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph
import networkx as nx

from tqdm import tqdm

from benji_src.benji_girgs.generation import get_dists
from benji_src.benji_girgs import utils
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
            std_distances = np.std(distances)
            # distances[distances > unique_dists[-2]] = -1

    if not fixed_infinite_dist:
        std_distances = np.std(distances)

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

        # print(a, b)
        new_a, new_b, seen_a, seen_b, dist_a, dist_b, met = utils.bi_bfs(g, a, b)
        sizes.append(len(seen_a) + len(seen_b))

    unique_sizes, sizes_counts = np.unique(sizes, return_counts=True)

    return unique_sizes, sizes_counts

    

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
