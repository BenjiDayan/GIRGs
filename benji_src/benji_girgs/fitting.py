from itertools import zip_longest
import networkit as nk
import networkx as nx
import numpy as np
from networkit.graph import Graph
import networkx as nx
import pandas as pd
import seaborn as sns

from tqdm import tqdm
import powerlaw

from benji_girgs.generation import cgirg_gen, get_dists, cgirg_gen_cube
from benji_girgs import utils
from benji_girgs.utils import avg_degree, get_perc_lower_common_nhbs

# import os
# if not "NO_CPP_GIRGS" in os.environ:
try:
    from girg_sampling import girgs
except Exception:
    pass

def scale_param(param, scale, base, larger=True, eps=1e-10):
    if larger:
        out = base + (param - base) * (1 - scale)**(-1)
    else:
        out = base + (param - base) * (1 - scale)
    return max(out, base + eps)  # make sure it's not too small - alpha=1.0 throws an error.

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
    if len(unique_dists) > 1:
        if unique_dists[-1] != unique_dists[-2] + 1:
            unique_dists[-1] = -1

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
    g = utils.get_largest_component(g)

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
    def __init__(self, target_avg_degree, girg_gen_func_const):
        super().__init__(
            target_avg_degree, 
            girg_gen_func_const, 
            metric_func=utils.avg_degree,
            metric_name='avgdeg',
            init_param=1.0,
            param_base=0.0,
            param_name='const')


class GirgMetricFitter(GirgFitter):
    def __init__(self, target_metric, girg_gen_func, metric_func, metric_name='avgdeg', init_param=1.0, param_base=0.0, param_name='const'):
        self.t = 0
        self.scale = 0.3
        # If metric doesn't overshoot, increase scale by 10%
        self.scale_up = 1.1
        # If metric overshoots, decrease scale by 50%
        self.scale_down = 0.5
        # the fit is considered good if the metric is within 3% of the target metric
        self.fit_percent = 0.03
        # After fit is achieved take at least this many more steps to confirm
        # The last step must be fitting or we repeat.
        self.post_fit_steps = 0
        self.max_fit_steps = 100 # max number of steps to take before giving up when self.fit

        self.fit_count = -1

        self.target_metric = target_metric
        self.girg_gen_func = girg_gen_func
        self.overshoot = True
        self.verbose=False
        self.param = init_param
        self.param_base = param_base
        self.param_name = param_name

        self.metric_func = metric_func
        self.metric_name = metric_name

        girg = self.girg_gen_func(self.param)
        self.metric = metric_func(girg)
        # avg degree too small -> need to make const larger
        self.larger = self.metric < self.target_metric

        self.hist = [(self.param, self.metric)]

    def step(self):
        self.param = scale_param(self.param, self.scale, self.param_base, self.larger)
        assert self.param >= self.param_base

        girg = self.girg_gen_func(self.param)
        metric = self.metric_func(girg)
        # e.g. 0.2 if mu too big
        larger = metric < self.target_metric

        self.overshoot = (larger != self.larger)
        self.scale *= (self.scale_down if self.overshoot else self.scale_up)
        self.scale = min(self.scale, 0.9)

        if self.verbose:
            print(f'{self.metric_name}: {self.metric:.3f} -> {self.metric_name}2: {metric:.2f}; overshoot: {self.overshoot}, scale:{self.scale:.3f}, {self.param_name}: {self.param:.3f}')

        self.larger = larger
        self.metric = metric

        self.hist.append((self.param, self.metric))

    def check_fit(self):
        did_fit = abs(self.metric - self.target_metric) / self.target_metric < self.fit_percent
        if did_fit and self.verbose:
            print(f'Fit! fit_count: {self.fit_count}; post_fit_steps: {self.post_fit_steps}')
        return did_fit

    def fit(self):
        """Steps until the average degree is within 3% of the target average degree
        Some extra steps are taken once the fit is acheived to confirm that the fit is stable.
        These extra steps may be repeated if the fit is not stable."""
        num_steps = 0
        while True:
            self.step()
            num_steps +=1 
            if num_steps > self.max_fit_steps:
                raise NotImplementedError('Max fit steps exceeded')
            now_fit = self.check_fit()
            if self.fit_count == -1:
                if now_fit:
                    self.fit_count = 0
            elif self.fit_count == self.post_fit_steps:
                if now_fit:
                    break
                else:
                    self.fit_count = 0
            else:
                self.fit_count += 1


def regularised_graph_distance(g, degree_percentile=0.9, n_samples=1000):
    dd = nk.centrality.DegreeCentrality(g).run().scores()
    dd = np.array(dd)
    bot_percentile_degree_indices = np.argsort(dd)[:int(degree_percentile*len(dd))]
    g_sub = utils.quick_subgraph(g, bot_percentile_degree_indices)
    cc = nk.components.ConnectedComponents(g_sub)
    # cc.run()
    g_sub = cc.extractLargestConnectedComponent(g_sub, True)
    if g_sub.numberOfNodes() < degree_percentile*0.15*len(dd):
        raise ValueError('Too many nodes removed')
    distances, std_distances, unique_dists, dist_counts = graph_distances(g_sub, n_samples)

    return distances, std_distances, unique_dists, dist_counts

# degree_percentile of e.g. <0.9 is quite important, otherwise there is barely any 
# difference in gdist - high degree nodes influence this too much.
# Decreasing this maximally is probably good, up to the point where it's too small.
# Ehhh. Maybe just keep as 0.9.
def regularised_std_graph_distance(g, degree_percentile=0.9, n_samples=1000):
    distances, std_distances, unique_dists, dist_counts = regularised_graph_distance(g, degree_percentile, n_samples)
    return std_distances

def regularised_median_graph_distance(g, degree_percentile=0.9, n_samples=1000):
    distances, std_distances, unique_dists, dist_counts = regularised_graph_distance(g, degree_percentile, n_samples)
    return np.median(distances)

def regularised_mean_graph_distance(g, degree_percentile=0.9, n_samples=1000):
    distances, std_distances, unique_dists, dist_counts = regularised_graph_distance(g, degree_percentile, n_samples)
    return np.mean(distances)

# TODO is this used/needed?
class GirgAlphaFitter(GirgMetricFitter):
    def __init__(self, target_std_gdist, target_avg_degree, girg_gen_func_alpha_const):
        """girg_gen_func should take alpha and const as arguments"""
        self.const = 1.0
        self.girg_gen_func_alpha_const = girg_gen_func_alpha_const
        self.target_avg_degree = target_avg_degree
        self.const_scale = 0.3
        self.const_hist = []

        super().__init__(target_std_gdist,
                                self.girg_gen_func,
                                metric_func=utils.LCC, 
                                metric_name='LCC', 
                                init_param=1.3,
                                param_base=1.0,
                                param_name='alpha')

    def girg_gen_func_const(self, const):
        return self.girg_gen_func_alpha_const(self.param, const)
    
    def fit_const(self):
        gcf = GirgMetricFitter(
            self.target_avg_degree, 
            self.girg_gen_func_const, 
            metric_func=utils.avg_degree,
            metric_name='avgdeg',
            init_param=self.const,
            param_base=0.0,
            param_name='const')
        # Pump it up again as with a new alpha things have shifted
        gcf.scale = min(self.const_scale * 1.5, 0.9)
        gcf.verbose=self.verbose
        gcf.fit()
        self.const = gcf.param
        self.const_scale = gcf.scale
        self.const_hist.append(gcf.hist)

    def girg_gen_func(self, alpha):
        self.fit_const()
        return self.girg_gen_func_alpha_const(alpha, self.const)

    
class CGirgAlphaFitter(GirgMetricFitter):
    def __init__(self, target_std_gdist, target_avg_degree, girg_and_const_gen_func):
        """girg_gen_func should take alpha and const as arguments"""
        self.const = 1.0
        self.girg_and_const_gen_func = girg_and_const_gen_func
        self.target_avg_degree = target_avg_degree

        super().__init__(target_std_gdist,
                                self.girg_gen_func,
                                metric_func=utils.LCC, 
                                metric_name='lcc', 
                                init_param=1.3,
                                param_base=1.0,
                                param_name='alpha')
        
    def girg_gen_func(self, alpha):
        girg, const = self.girg_and_const_gen_func(alpha)
        self.const = const
        if self.verbose:
            print(f'const: {self.const:.3f}')
        return girg
    
    
def fit_cgirg(g, d, fit_percent=0.04, max_fit_steps=13, post_fit_steps=2, verbose=False):
    """Fit a CGIRG to a given graph g. Returns alpha, const, hist,
    and True/False whether the fit was successful. Hopefully even if the fit
    was not successful, the returned alpha and const will be close to the
    a good value?"""
    tau = utils.powerlaw_fit_graph(g)


    target_avg_degree = utils.avg_degree(g)
    # try:
    #     target_std_gdist = regularised_std_graph_distance(g)
    # except ValueError as e:
    #     print(e)
    #     alpha = 1.3
    #     weights = girgs.generateWeights(g.numberOfNodes(), tau)
    #     const = girgs.scaleWeights(weights, target_avg_degree, d, alpha)
    #     return alpha, const, tau, [], False

    target_lcc = utils.LCC(g)


    if verbose:
        print(f'target_avg_degree: {target_avg_degree:.3f}')
        print(f'target_lcc: {target_lcc:.3f}')
        print(f'fit tau: {tau:.3f}')

    n = g.numberOfNodes()


    def girg_and_const_gen_func(alpha):
        g, edges, weights, pts, c, id2gnk = cgirg_gen(n, d, tau, alpha, desiredAvgDegree=target_avg_degree, weights=None)
        return g, c

    caf = CGirgAlphaFitter(target_lcc, target_avg_degree, girg_and_const_gen_func)
    # Try to be quite lenient in allowing fitting to occur :)
    caf.fit_percent = fit_percent
    caf.max_fit_steps = max_fit_steps
    caf.post_fit_steps = post_fit_steps
    caf.verbose = verbose
    try:
        caf.fit()
    except (NotImplementedError, ValueError) as e:
        print(e)
        return caf.param, caf.const, tau, caf.hist, target_lcc, False
    return caf.param, caf.const, tau, caf.hist, target_lcc, True




def gp_girg_cube_fitter(g, d, tau, n_calls=30, base_estimator=None):
    target_lcc = utils.LCC(g)
    target_avg_degree = utils.avg_degree(g)
    n = g.numberOfNodes() * 2 ** d

    weights = girgs.generateWeights((2 ** d) * n, tau)
    const_guess = girgs.scaleWeights(weights, target_avg_degree, d, alpha) * 1.3

    def fun_to_optimise(params):
        alpha = 1 / params[0]
        const = np.exp(params[1]) * const_guess
        g, _ = generation.cgirg_gen_cube(n, d, tau, alpha, const=const)
        a, b = utils.LCC(g), target_lcc
        x, y = utils.avg_degree(g), target_avg_degree
        return np.log(a / b) ** 2 + 10 * np.log(x / y) ** 2

    result = gp_minimize(fun_to_optimise, dimensions=[(0.01, 0.99), (-1.4, 1.4)], n_calls=n_calls, verbose=False,
                         noise="gaussian", base_estimator=base_estimator)
    return result

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
        self.p_uv = np.divide(outer, self.dists**self.d)

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

