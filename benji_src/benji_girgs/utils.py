from itertools import zip_longest
import networkit as nk
import numpy as np
import scipy
from networkit.graph import Graph
import networkx as nx
import sys
import os
import powerlaw
from sklearn.neighbors import KDTree

# import os
# if not "NO_CPP_GIRGS" in os.environ:
try:
    from girg_sampling import girgs
except Exception:
    pass

from benji_girgs import points

def avg_degree(g: Graph):
    num_edges = g.numberOfEdges()
    num_nodes = g.numberOfNodes()
    return (2 * num_edges) / num_nodes

def LCC(g):
    """local clustering coefficient - average of CC over all nodes.
    Same as given by the nk.overview
    nk.globals.clustering(g) gives an approximation of the local clustering coefficient
    """
    lcc = nk.centrality.LocalClusteringCoefficient(g)
    lcc.run()
    return np.mean(lcc.scores())

def graph_degrees_to_weights(g: Graph):
    """Converts a graph's degrees to weights, and returns the weighted graph.
    """
    degrees = nk.centrality.DegreeCentrality(g).run().scores()
    return degrees

def get_top_k_clique(g: Graph, k=10):
    degrees = graph_degrees_to_weights(g)
    argsorted = np.argsort(degrees)
    g2 = quick_subgraph(g, argsorted[-k:])
    return g2

def empirical_graph_top_k(g):
    degrees = graph_degrees_to_weights(g)
    argsorted = np.argsort(degrees)
    outs = []
    W = np.sum(degrees)
    for k in range(2, 20):
        a, b = degrees[argsorted[-k]], degrees[argsorted[-k+1]]
        outs.append((k, a, b, a*b/W))

    return outs



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def powerlaw_fit_graph(g):
    dd = sorted(nk.centrality.DegreeCentrality(g).run().scores(), reverse=True)
    with HiddenPrints():
        fit = powerlaw.Fit(dd, discrete=True)
    return fit.power_law.alpha


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

        a_deg, b_deg = g.degree(a), g.degree(b)
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



# TODO update as simple_bfs func has changed
def simple_bfs_search(g, weights, weight_thresh):
    """Performs bfs search """
    suitable_starting = np.argwhere(weights < weight_thresh).reshape(-1)
    dist_2s = []
    for starting_node in suitable_starting[:10]:
        dists = simple_bfs(g, weights, weight_thresh, starting_node)
        dists2 = {k: len(v) for k, v in dists.items()}
        dist_2s.append(dists2[2])
    return np.median(dist_2s)


def simple_bfs(g, starting_node, degree_thresh=None):
    """BFS out from starting_node, only accessing nodes with weight < weight_thresh.
    Nodes with higher weights are ignored.

    return: dist = {0: [starting_node], 1: [nodes of dist 1], 2: [nodes of dist 2], ...}"""
    dists = {0: set([starting_node])}
    visited = set([starting_node])
    dist = 0
    while True:
#         print(dist)
        trigger = False
        new_dist = set([])
        for node in dists[dist]:
            for nhb in g.iterNeighbors(node):
                if not nhb in visited:
                    trigger = True
                    visited.add(nhb)
                    if degree_thresh is not None:
                        if g.degree(nhb) < degree_thresh:
                            new_dist.add(nhb)
                    else:
                        new_dist.add(nhb)

        dist += 1
        dists[dist] = new_dist
        if not trigger:
            break

    return dists

def bi_bfs(g, a, b):
    dists_a = simple_bfs(g, a)
    dists_b = simple_bfs(g, b)

    dist_a = 0
    dist_b = 0
    seen_a = set()
    seen_b = set()
    while True:
        if dist_a in dists_a:
            new_a = dists_a[dist_a]
            dist_a += 1
        else:
            new_a = set()

        if dist_b in dists_b:
            new_b = dists_b[dist_b]
            dist_b += 1
        else:
            new_b = set()

        if new_a == set() and new_b == set():
            return new_a, new_b, seen_a, seen_b, dist_a -1, dist_b -1, False

        if new_a.intersection(seen_b) or new_b.intersection(seen_a) or new_a.intersection(new_b):
            break
        seen_a.update(new_a)
        seen_b.update(new_b)

    return new_a, new_b, seen_a, seen_b, dist_a -1, dist_b -1, True



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
    cc = nk.components.ConnectedComponents(g)
    cc.run()
    return cc.extractLargestConnectedComponent(g, True)


def get_diffmap_old(g, Iweighting=0.5, sparse_evals=10, **kwargs):
    """
    Provided g is connected, find the diffusion map of g.
    w are the eigenvalues, decreasing from 1.0, lambda_2, lambda_3, ...
    diff_map(i) is the tth diffusion iteration of node i as a linear combination of Psi columns

    eye_or_ones is experimental. Bandeira uses identity (eye), but I wonder if ones might do
    a better job here?

    sparse_evals: We do things sparsely, and the sparse eigenvalue finder can actually
    restrict to the top k eigenvalues, so we will use this. As long as you only e.g. care about
    max the top 3 dimensions, happily set this to 10 I guess.
    """
    if nk.components.ConnectedComponents(g).run().numberOfComponents() > 1:
        raise Exception("Graph is not connected")


    gnx = nk.nxadapter.nk2nx(g)

    # A = nx.linalg.adjacency_matrix(gnx).todense()
    A = nx.linalg.adjacency_matrix(gnx)

    D = np.array([x[1] for x in (gnx.degree)])
    D_h = D**(0.5)
    D_hi = D**(-0.5)

    n = A.shape[0]

    # M_ij = W_ij / d_i
    # M = np.diag(1/D) @ A
    # sparse array version
    M = (A.T).multiply(1/D).T

    # M = (1-Iweighting)* M + Iweighting * np.eye(M.shape[0])
    # M = (1 - Iweighting) * M + Iweighting * (np.ones((n, n))/n if eye_or_ones == "ones" else np.eye(n))
    # sparse array version
    M = (1 - Iweighting) * M + scipy.sparse.eye(n) * Iweighting

    # S_ij = W_ij / sqrt(d_i d_j) = sqrt(d_i) M_ij / sqrt(d_j)
    # S = np.diag(D_h) @ M @ np.diag(D_hi)
    # sparse array version
    S = scipy.sparse.diags(D_h) @ M @ scipy.sparse.diags(D_hi)

    # w, V = np.linalg.eigh(S)
    # sparse array version
    w, V = scipy.sparse.linalg.eigsh(S, k=sparse_evals, which="LM")

    # S = V @ np.diag(w) @ V.T
    # M = D^{-1/2} @ S @ D^{1/2}
    # = D^{-1/2} @ V @ np.diag(w) @ V.T @ D^{1/2}
    # = Phi @ np.diag(w) @ Psi.T
    # Phi = D^{-1/2} @ V
    # Psi = D^{1/2} @ V
    # Phi = np.diag(D_hi) @ V
    # Psi = np.diag(D_h) @ V
    # sparse array version
    Phi = scipy.sparse.diags(D_hi) @ V
    Psi = scipy.sparse.diags(D_h) @ V

    w = np.flip(w)

    def diff_map(i, t):
        """
        diff map of v_i is (l_1^t phi_1(i), l_2^t phi_2(i), ..., l_n^t phi_n(i))
        phi_1(i) = Phi[i, 0],
        phi_2(i) = Phi[i, 1] etc.

        Hence we want Phi[i, n-1], Phi[i, n-2], ..., Phi[i, 0] because actually we flipped the eigenvalues to
        be largest at the front.

        Returns the diffusion map of node i after t iterations, ordered from largest eigenvalue to smallest.
        """
        # n = 5, so 0, 1, 2, 3, 4, we want to get 3, 2, 1, 0
        # so 5-2 -> -1, -1
        # return np.array([Phi[i, j] for j in range(n-2, -1, -1)]) * (w[1:]**t)
        return Phi[i, -2::-1] * (w[1:] ** t)

    return w, Phi, Psi, diff_map

def get_diffmap(g, Iweighting=0.5,  sparse_evals=10, gamma=0.9, heat_kernel=0, **kwargs):
    """
    Provided g is connected, find the diffusion map of g.
    w are the eigenvalues, decreasing from 1.0, lambda_2, lambda_3, ...
    diff_map(i) is the tth diffusion iteration of node i as a linear combination of Psi columns

    eye_or_ones is experimental. Bandeira uses identity (eye), but I wonder if ones might do
    a better job here?

    sparse_evals: We do things sparsely, and the sparse eigenvalue finder can actually
    restrict to the top k eigenvalues, so we will use this. As long as you only e.g. care about
    max the top 3 dimensions, happily set this to 10 I guess.

    heat_kernel?? wtf. See Belkin Nyiogi Laplacian Eigenmaps for Dimensionality Reduction and Data
    Representation
    """
    if nk.components.ConnectedComponents(g).run().numberOfComponents() > 1:
        raise Exception("Graph is not connected")


    gnx = nk.nxadapter.nk2nx(g)

    # A = nx.linalg.adjacency_matrix(gnx).todense()
    A = nx.linalg.adjacency_matrix(gnx)

    if heat_kernel:
        d = heat_kernel  # integer??
        A = A.astype(np.float32)
        # if u~v then E[r_uv] is about such that p_uv=1, i.e. r_uv = (w_u w_v/n)^(1/d)
        # Then heat kernel thing says that we want W_ij = exp(-r_ij^2 / T) for some param T
        dd = nk.centrality.DegreeCentrality(g).run().scores()
        n = g.numberOfNodes()
        T = 1.0 # ??
        for i, j in g.iterEdges():
            r_ij = (dd[i] * dd[j] / n)**(1/d)
            A[i, j] = np.exp(-r_ij**2 / T)
            A[j, i] = A[i, j]

        D = A.sum(axis=1)

    else:
        D = np.array([x[1] for x in (gnx.degree)])
    D_h = D**(0.5)
    D_hi = D**(-0.5)

    n = A.shape[0]


    # Empirically this gamma seems to work well.
    # It discourages taking edges to popular nhbs.
    M_tilde = scipy.sparse.diags(1 / D) @ A @ scipy.sparse.diags(D ** (-gamma))
    M_tilde = scipy.sparse.diags(np.array(1 / M_tilde.sum(axis=-1)).squeeze()) @ M_tilde
    M_tilde = (1 - Iweighting) * M_tilde + Iweighting * np.eye(M_tilde.shape[0])
    a, B = scipy.sparse.linalg.eigs(M_tilde, k=sparse_evals, which="LR")
    # I think we should take absolute values instead?
    # Although it does seem like everything is real anyway. TODO why?
    a, B = np.real(a), np.real(B)

    idx = np.argsort(a)[::-1]
    a = a[idx]
    B = B[:, idx]

    def diff_map(i, t):
        return B[i, 1:] * a[1:] ** t


    return a, B, None, diff_map

# This function is quite good at fixing diffmap points to be more square like.
# There is sometimes an issue whereby e.g. in the 2D case, fixing the 2nd dim,
# we get an X.
# I generally think it should only be used on cuboid GIRGs, since if the diffmap
# isn't at heart somewhat cuboid, it's really shifty
def cubify_dim(pts, k=1, perc_near=0.05, quantile=None):
    n = pts.shape[0]
    # only looks at dimensions 0:k
    pts_without_k = np.concatenate([pts[:, 0:k], pts[:, k+1:]], axis=-1)
    pts_kdtree = KDTree(pts_without_k)
    output = []

    # This is a crude filter - we only look at the 5% nearest points. We hope that the smoothness
    # of the map is roughly ok in this range
    num_near = int(pts.shape[0] * perc_near)
    for u in range(n):
        distances, indices = pts_kdtree.query(pts_without_k[u:u+1], k=num_near)
        second_dim = pts[indices, k].squeeze()
        u_second_dim = pts[u, k]
        scale = 1.0 if not quantile else np.quantile(second_dim, 1 - quantile) - np.quantile(second_dim, quantile)
        output.append(scale * scipy.stats.percentileofscore(second_dim, u_second_dim)/100)
    return output

# output = cubify_last_k_dim(pts_temp)
# pts_temp_fixed = pts_temp.copy()
# pts_temp_fixed[:, 1] = output

# WTF is going on here? I'm confused...


# uniformify is probably best for real life graphs
# cubify best for true scale GIRGs
# cuboidify might be better for preserving distorted scale?
def get_diffmap_and_points(g, spare_evals=10, ds=2, process=None):
    a, B, _, diff_map = get_diffmap(g, sparse_evals=spare_evals)
    pts = np.array([diff_map(i, 10) for i in range(g.numberOfNodes())])
    pts = pts[:, :ds]
    if process == 'uniformify':
        pts = uniformify_pts(pts)
    elif process == 'cubify':
        # so if ds=2 then just [1], if ds=3 then [2, 1], ...
        outputs = []
        for k in range(0, ds):
            output = cubify_dim(pts, k=k, perc_near=0.1)
            outputs.append(output)
        pts = np.stack(outputs, axis=-1)
        # pts[:, 1:] = np.concatenate(outputs, axis=-1)
        # pts[:, 0] = percentileify(pts[:, 0])
    elif process == 'cuboidify':
        outputs = []
        for k in range(0, ds):
            output = cubify_dim(pts, k=k, quantile=0.05, perc_near=0.1)
            # pts[:, k] = output
            outputs.append(output)
        pts = np.stack(outputs, axis=-1)
        # pts[:, 0] = percentileify(pts[:, 0], 0.05)
    elif process == 'put_in_cube':
        pts = points.normalise_points_to_cube(pts)
    elif process == 'restrict_uniform_edges':
        pts = restrict_and_uniformify_edges(pts, 0.05)
    return a, B, pts


def percentileify(arr, quantile=None):
    """Given a flat array, return the percentile of each element.

     e.g. [0.1, 0.7, 0.2, 0.3, 0.6, 0.35]
        ->   [0, 2, 3, 5, 4, 1]
        ->   [0, 5, 1, 2, 4, 3]
        i.e. 0.1 is 0th, 0.7 is 5th, 0.2 is 1st, 0.3 is 2nd, 0.6 is 4th, 0.35 is 3rd
    """
    if quantile is None:
        scale = 1.0
    else:
        # quantile should be e.g. 0.1
        scale = np.quantile(arr, 1 - quantile) - np.quantile(arr, quantile)
    return scale * (arr.argsort().argsort() / arr.size)
def uniformify_pts(pts: np.ndarray):
    """Will uniformlyish distribute the points in the unit cube."""
    n, d = pts.shape
    pts_out = np.zeros_like(pts)
    for i in range(d):
        # e.g. [0.1, 0.7, 0.2, 0.3, 0.6, 0.35]
        # ->   [0, 2, 3, 5, 4, 1]
        # ->   [0, 5, 1, 2, 4, 3]
        # i.e. 0.1 is 0th, 0.7 is 5th, 0.2 is 1st, 0.3 is 2nd, 0.6 is 4th, 0.35 is 3rd
        pts_out[:, i] = percentileify(pts[:, i])
    return pts_out

def restrict_to_quantiles(pts, quantile=0.05):
    foo = np.ones(pts.shape[0])
    for d in range(pts.shape[1]):
        p1, p2 = np.quantile(pts[:, d], (quantile, 1-quantile))
        foo = foo * (pts[:, d] > p1) * (pts[:, d] < p2)

    return pts[foo.astype(bool), :], foo.astype(bool)

def restrict_and_uniformify_edges(pts, quantile=0.05):
    """centres the central quantile to a 0.05 - 0.95 cube, and then uniformly distributes the rest."""
    pts_new = pts.copy()
    pts_restricted, restriction = restrict_to_quantiles(pts, quantile=quantile)
    pts_restricted = points.normalise_points_to_cube(pts_restricted) * 0.9 + 0.05
    pts_new[restriction, :] = pts_restricted

    n, d = pts.shape

    for i in range(d):
        small = pts[:, i] < np.quantile(pts[:, i], quantile)
        big = pts[:, i] > np.quantile(pts[:, i], 1-quantile)
        pts_new[small, i] = (pts_new[small, i].argsort().argsort() / small.sum()) * 0.05
        pts_new[big, i] = 0.95 + (pts_new[big, i].argsort().argsort() / big.sum()) * 0.05

    return pts_new




# gnx = nk.nxadapter.nk2nx(g)
# A = nx.linalg.adjacency_matrix(gnx).todense()
#
# D = np.array([x[1] for x in (gnx.degree)])
#
# M = np.diag(1/D) @ A
# # M = (1-Iweighting)* M + Iweighting * np.eye(M.shape[0])
# M = (1 - 0.5) * M + 0.5 * np.eye(n)
#
# l, V = np.linalg.eig(M)
#
# def diff_map(i, t):
#     # n = 5, so 0, 1, 2, 3, 4, we want to get 3, 2, 1, 0
#     # so 5-2 -> -1, -1
#     return np.array([V[i, j] for j in range(n-2, -1, -1)]) * (l[1:]**t)
#
# pts = np.array([diff_map(i, 10) for i in range(g.numberOfNodes())])
# pts.shape
# plt.figure()
# xs = pts[:, 0]
# ys = pts[:, 1]
# plt.figure()
# plt.scatter(xs, ys)
#
# plt.figure()
# plt.scatter(xs, pts_torus[:, 1])
#
# plt.scatter(ys, pts_torus[:, 0])

##########################

# w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.1, eye_or_ones='eye')
#
# gnx = nk.nxadapter.nk2nx(g)
# A = nx.linalg.adjacency_matrix(gnx).todense()
#
# D = np.array([x[1] for x in (gnx.degree)])
# D_h = D**(0.5)
# D_hi = D**(-0.5)
#
# M = np.diag(1/D) @ A
# theta = 0.1
# M_tilde = M @ np.diag(D**(-theta))
# K = M_tilde.sum(axis=1)
# K = 1/K
# M_tilde = np.diag(K) @ M_tilde
# M_tilde = 0.9* M_tilde + 0.1 * np.eye(M_tilde.shape[0])
#
# # M_tilde = 0.5 * M_tilde + 0.5 * M_tilde.T
#
# # w, Phi = np.linalg.eigh(M_tilde)
#
# S = np.diag(1/K) @ np.diag(D_h) @ M_tilde @ np.diag(D_hi)
# w,  V = np.linalg.eigh(S)
# Phi = np.diag(D_hi) @ V
# Psi = np.diag(D_h) @ V
#
# n = Phi.shape[0]
# w = np.flip(w)
#
# # # Important bit!!!
# Phi = np.diag(K) @ Phi
#
# def diff_map(i, t):
#     # n = 5, so 0, 1, 2, 3, 4, we want to get 3, 2, 1, 0
#     # so 5-2 -> -1, -1
#     return np.array([Phi[i, j] for j in range(n-2, -1, -1)]) * (w[1:]**t)
#
#
#
# ys = w[1:13]**8
# #     plt.figure()
# _ = plt.scatter(list(range(1, len(ys)+1)), ys, label=f'b: {b}')
# #     plt.show()
# plt.legend()
# plt.show()
#
#
# pts = np.array([diff_map(i, 10) for i in range(n)])
# pts.shape
# plt.figure()
# xs = pts[:, 0]
# ys = pts[:, 1]
# plt.figure()
# plt.scatter(xs, ys)
#
# fig, axes = plt.subplots(2, figsize=(5, 10))
# axes[0].scatter(xs, pts_torus[:, 1])
# axes[1].scatter(ys, pts_torus[:, 0])