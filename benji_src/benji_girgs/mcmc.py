import pickle
from unittest.mock import MagicMock

import networkit as nk
import networkx as nx
import numpy as np
import scipy
from sklearn.neighbors import KDTree

from benji_girgs import points, generation, utils
import multiprocessing
import matplotlib.pyplot as plt
import ctypes
from tqdm import tqdm

import seaborn as sns
import os


class MCMC_girg():
    def __init__(self, g: nk.Graph, weights: np.ndarray, alpha: float, const: float, pts: points.PointsCube,
                 pool=False, graph_name=None):
        """
        Does a 1D MCMC fit for the cube GIRG model
        weights should be the degree sequence
        alpha, const should have already been estimated from GIRG fitting using these weights
        """
        self.n = g.numberOfNodes()
        self.d = pts.shape[1]

        # # We use diffusion map for an initial pts estimate
        # w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.5, eye_or_ones='eye')
        # pts_diffmap = np.array([diff_map(i, 10) for i in range(self.n)])
        # pts_diffmap = points.normalise_points_to_cube(pts_diffmap[:, 0:1])
        # pts_diffmap = points.PointsCube(pts_diffmap)
        # self.pts = pts_diffmap

        self.pool = pool
        self.pts_type = type(pts)
        if pool:
            self.shared_pts = multiprocessing.Array(ctypes.c_double, pts.flatten())
            # self.pts = self.pts_type(np.frombuffer(self.shared_pts).reshape(pts.shape))
            self.pts = self.pts_type(np.frombuffer(self.shared_pts.get_obj()).reshape(pts.shape))
            # we could use this as a prior?
            self.pts_init = pts.copy()
            self.shared_pts_init = multiprocessing.Array(ctypes.c_double, self.pts_init.flatten())
            # is it necessary to redo this?
            self.pts_init = self.pts_type(np.frombuffer(self.shared_pts_init.get_obj()).reshape(pts.shape))

        else:
            self.pts = pts


        # Will this work??
        global fitting_graph
        fitting_graph = g
        self.g = g
        self.graph_name = graph_name

        self.weights = weights / np.sqrt(weights.sum())
        self.weights_original = weights
        self.const_in = generation.const_conversion(const, alpha, d=self.d, true_volume=True)
        self.const = const
        self.alpha = alpha
        # self.p_uv = generation.get_probs(weights / np.sqrt(weights.sum()), self.pts, alpha, self.const_in)
        # self.u_lls = []


        print('calibrating const')
        for _ in range(7):
            self.ll, self.expected_num_edges = self.calculate_ll()
            print(f'const: {self.const}, expected_num_edges: {self.expected_num_edges}')
            self.calibrate_const()

        self.ll_steps = [0]
        self.lls = [self.ll]
        self.num_acceptances = 0
        self.num_steps = 0


        gnx = nk.nxadapter.nk2nx(g)
        self.A = nx.adjacency_matrix(gnx).todense()

        g_out, A_out, out, percent_edges_captured, percent_fake_edges_wrong = self.get_CM(self.A)
        self.percent_edges_captureds = [(percent_edges_captured, 0)]
        self.outs = [(out, 0)]

        # if pool:
        #     self.pts_old = self.pts.copy()
        #     self.pts = multiprocessing.RawArray(ctypes.c_double, self.pts.flatten())

        # if pool:
        #     self.pool = multiprocessing.Pool(pool)
        #     self.shared_pts = multiprocessing.RawArray(ctypes.c_double, int(np.prod(self.pts.shape)))
        #     self.pts_old = self.pts.copy()
        #     self.pts = np.ndarray(self.pts.shape, dtype=np.float64, buffer=self.shared_pts)
        #     np.copyto(self.pts, self.pts_old)
        #
        #     self.ll_steps = multiprocessing.Queue()
        #     self.lls = multiprocessing.Queue()
        #     self.ll = multiprocessing.RawValue(ctypes.c_double, self.ll)
        #     self.lls.put(self.ll)
        #     self.ll_steps.put(0)
        #     self.num_acceptances = multiprocessing.RawValue(ctypes.c_int, self.num_acceptances)
        #
        # else:
        #     self.pool = None

    def calibrate_const(self):
        """Makes self.const, self.const_in smaller or bigger to match the desired number of edges
        We use this random e^+-a to make life easier
        We should really use a MCMC log likelihood approach instead but this is a bit of a simpler hack.

        Actually a better hack we'll use this ratio trick :/ why not"""
        # bigger = self.expected_num_edges < self.g.numberOfEdges()
        # a = np.random.uniform()
        # if bigger:
        #     self.const = self.const * np.exp(a)
        # else:
        #     self.const = self.const * np.exp(-a)
        ratio = self.expected_num_edges / self.g.numberOfEdges()
        self.const = self.const/ratio
        self.const_in = generation.const_conversion(self.const, self.alpha, d=self.d, true_volume=True)
    def calculate_ll(self):
        """NB our iterative updating of self.ll is rather approximate, so would need to be
        recalculated from scratch periodically if we wanted to be more accurate"""
        # This is going to be double the actual LL as we double count edges
        ll = 0
        expected_num_edges = 0
        for u_index in range(self.n):
            eps = 1e-7
            p_u_to_vs = generation.get_probs_u(self.weights, self.pts, self.alpha, self.const_in, u_index)
            expected_num_edges += p_u_to_vs.sum()
            p_u_to_vs = np.clip(p_u_to_vs, eps, 1 - eps)
            u_ll = self.p_u_to_vs_to_ll(self.g, u_index, p_u_to_vs)
            ll += u_ll
        return ll, expected_num_edges/2

    @staticmethod
    def p_u_to_vs_to_ll(g, u_index, p_u_to_vs):
        """p_u_to_vs is a vector of probabilities of u to each v"""
        out = 0
        n = g.numberOfNodes()
        assert n == len(p_u_to_vs)
        mask = np.ones(n, dtype=bool)
        for nhb in g.iterNeighbors(u_index):
            mask[nhb] = False
            out += np.log(p_u_to_vs[nhb])
        mask[u_index] = False

        out += np.log(1 - p_u_to_vs[mask]).sum()
        return out

    def update_ll(self, u_index, p_u_to_vs_old, p_u_to_vs_new):
        """
        In the adjacency matrix, we have a row u -> :, and then a column : -> u, which
        intersect at u -> u (which is ignored anyways).

        Hence we update the log likelihood twice, once for the row, once for the column.
        Args:
            u_index:
            p_u_to_vs_old:
            p_u_to_vs_new:
        """
        u_ll_old = self.p_u_to_vs_to_ll(self.g, u_index, p_u_to_vs_old)
        u_ll_new = self.p_u_to_vs_to_ll(self.g, u_index, p_u_to_vs_new)
        self.ll += 2 * (u_ll_new - u_ll_old)

    @staticmethod
    def acceptance_prob_static(g, weights, alpha, const_in, pts, u_index, x_u2, prior_x_u=None):
        eps = 1e-7
        p_u_to_vs = generation.get_probs_u(weights, pts, alpha, const_in, u_index)
        p_u_to_vs = np.clip(p_u_to_vs, eps, 1 - eps)

        x_u = pts[u_index].copy()
        pts[u_index] = x_u2
        p_u_to_vs2 = generation.get_probs_u(weights, pts, alpha, const_in, u_index)
        pts[u_index] = x_u
        p_u_to_vs2 = np.clip(p_u_to_vs2, eps, 1 - eps)

        u_ll_old = MCMC_girg.p_u_to_vs_to_ll(g, u_index, p_u_to_vs)
        u_ll_new = MCMC_girg.p_u_to_vs_to_ll(g, u_index, p_u_to_vs2)

        if prior_x_u is not None:
            u_ll_old += np.log(prior_x_u(x_u))
            u_ll_new += np.log(prior_x_u(x_u2))

        Q_ratio = np.exp(u_ll_new - u_ll_old)
        acceptance_prob = min(1, Q_ratio)
        return acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs, p_u_to_vs2

    # @staticmethod
    # def acceptance_probs_static(g: nk.Graph, weights: np.ndarray, alpha: float,
    #                             const_in: float, pts: np.ndarray, u_indices: np.ndarray, x_u2: np.ndarray):
    #     """
    #
    #     Args:
    #         g:
    #         weights:
    #         alpha:
    #         const_in:
    #         pts:  (n, d)
    #         u_index: of size k
    #         x_u2: of size (k, d)
    #
    #     Returns:
    #
    #     """
    #     eps = 1e-7
    #     p_u_to_vs = generation.get_probs_u(weights, pts, alpha, const_in, u_index)
    #     p_u_to_vs = np.clip(p_u_to_vs, eps, 1 - eps)
    #
    #     x_u = pts[u_index].copy()
    #     pts[u_index] = x_u2
    #     p_u_to_vs2 = generation.get_probs_u(weights, pts, alpha, const_in, u_index)
    #     pts[u_index] = x_u
    #     p_u_to_vs2 = np.clip(p_u_to_vs2, eps, 1 - eps)
    #
    #     u_ll_old = MCMC_girg.p_u_to_vs_to_ll(g, u_index, p_u_to_vs)
    #     u_ll_new = MCMC_girg.p_u_to_vs_to_ll(g, u_index, p_u_to_vs2)
    #
    #     Q_ratio = np.exp(u_ll_new - u_ll_old)
    #     acceptance_prob = min(1, Q_ratio)
    #     return acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs, p_u_to_vs2

    def acceptance_prob(self, u_index, x_u2):
        return self.acceptance_prob_static(self.g, self.weights, self.alpha, self.const_in, self.pts, u_index, x_u2)

    def step0(self):
        u_index = np.random.randint(self.n)
        x_u2 = np.random.uniform(size=self.d)
        acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new = self.acceptance_prob(u_index, x_u2)
        return acceptance_prob, u_index, x_u2, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new

    # def step0_pool(self):
    #     global pts
    #     my_pts = self.pts_type(np.frombuffer(pts).reshape(self.n, self.d))
    #     u_index = np.random.randint(self.n)
    #     x_u2 = np.random.uniform(size=self.d)
    #     acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new = self.acceptance_prob(u_index, x_u2)
    #     return acceptance_prob, u_index, x_u2, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new

    def step1(self):
        acceptance_prob, u_index, x_u2, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new = self.step0()
        if np.random.rand() < acceptance_prob:
            self.ll += 2 * (u_ll_new - u_ll_old)
            self.pts[u_index] = x_u2
            return 1, u_index, x_u2
        else:
            return 0, u_index, x_u2

    def step2(self):
        accepted, u_index, x_u2 = self.step1()
        self.num_steps += 1
        if accepted:
            self.num_acceptances += 1
            self.lls.append(self.ll)
            self.ll_steps.append(self.num_steps)

    def pool_step(self, _):
        np.random.seed(seed=_)
        acceptance_prob, u_index, x_u2, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new = self.step0()
        accepted = np.random.rand() < acceptance_prob
        delta_ll = 2 * (u_ll_new - u_ll_old)
        return accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2

    def step_pool(self, i, pool_size=3, jobs_per_worker=5, plot_every=None, verbose=False):
        seeds = (self.num_steps * (jobs_per_worker * pool_size) + np.arange(jobs_per_worker * pool_size)) % (2**31)
        outputs = []
        with multiprocessing.pool.ThreadPool(pool_size) as p:
            results = p.imap_unordered(self.pool_step, seeds, chunksize=jobs_per_worker)
            for accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2 in results:
                if verbose:
                    print(f'acceptance_prob: {acceptance_prob}')
                    print(u_index)
                    print(x_u2)
                    print(acceptance_prob)
                    print(f'accepted: {accepted}')
                if accepted:
                    outputs.append((delta_ll, u_index, x_u2))

                self.num_steps += 1

            p.close()
            p.join()

            for delta_ll, u_index, x_u2 in outputs:
                self.ll += delta_ll
                self.pts[u_index] = x_u2
                self.num_acceptances += 1
                self.lls.append(self.ll)
                self.ll_steps.append(self.num_steps)


    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     del self_dict['pts']
    #     return self_dict

    def run(self, n_steps, plot_every=None, pool_size=None, jobs_per_worker=5):
        start_steps = self.num_steps
        with tqdm(total=n_steps) as pbar:
            while self.num_steps < n_steps + start_steps:
                if pool_size:
                    self.step_pool(self.num_steps, pool_size, jobs_per_worker)
                else:
                    self.step2()
                    self.num_steps += 1

                if type(plot_every) is int and self.num_steps % plot_every == 0:
                    self.plot_ll(n_steps)

                pbar.update(self.num_steps - start_steps - pbar.n)

    # another attempt with pool not thread pool. Pickling overhead! But what is pickled? Hopefully not too much.
    # def run_pool(self, n_steps, pool_size=5, verbose=False):
    #     with tqdm(total=n_steps) as pbar:
    #         while self.num_steps < n_steps:
    #             jobs_per_worker = 10
    #             seeds = self.num_steps*(jobs_per_worker*pool_size) + np.arange(jobs_per_worker*pool_size)
    #             with multiprocessing.Pool(processes=pool_size, initializer=mcmc_girg_init_worker, initargs=(self.pts, self.pts.shape, self.weights, self.g)) as p:
    #                 results = p.imap_unordered(mcmc_girg_pool_step, [(seed, self.alpha, self.const_in) for seed in seeds], chunksize=jobs_per_worker)
    #                 for accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2 in results:
    #                     if verbose:
    #                         print(f'acceptance_prob: {acceptance_prob}')
    #                         print(u_index)
    #                         print(x_u2)
    #                         print(acceptance_prob)
    #                         print(f'accepted: {accepted}')
    #                     if accepted:
    #                         break
    #                     else:
    #                         self.num_steps += 1
    #
    #                 p.terminate()
    #                 p.join()
    #
    #                 if accepted:
    #                     self.num_steps += 1
    #                     self.ll += delta_ll
    #                     pts = np.frombuffer(self.pts).reshape(self.n, self.d)
    #                     pts[u_index] = x_u2
    #                     self.num_acceptances += 1
    #                     self.lls.append(self.ll)
    #                     self.ll_steps.append(self.num_steps)
    #             pbar.update(self.num_steps - pbar.n)


    # TODO pool size doesn't speed this up enough??
    def run_pool(self, n_steps, pool_size=5, jobs_per_worker=5, plot_every=None, verbose=False,
                 sigma=None, p_normal=None, use_tqdm=True, save_plot=False):
        start_steps = self.num_steps
        prev_plot_step = 0
        with (tqdm(total=n_steps) if use_tqdm else MagicMock()) as pbar:
            while self.num_steps < n_steps + start_steps:
                seeds = (self.num_steps*(jobs_per_worker*pool_size) + np.arange(jobs_per_worker*pool_size)) % (2**31)
                outputs = []
                with multiprocessing.Pool(processes=pool_size, initializer=mcmc_girg_init_worker, initargs=(self.shared_pts, self.pts.shape, self.shared_pts_init, self.weights, self.g)) as p:
                    results = p.imap_unordered(mcmc_girg_pool_step, [(seed, self.alpha, self.const_in, sigma, p_normal) for seed in seeds], chunksize=jobs_per_worker)
                    for accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2 in results:
                        if verbose:
                            print(f'acceptance_prob: {acceptance_prob}')
                            print(u_index)
                            print(x_u2)
                            print(acceptance_prob)
                            print(f'accepted: {accepted}')
                        if accepted:
                            outputs.append((self.num_steps, delta_ll, u_index, x_u2))
                        self.num_steps += 1

                    p.close()
                    p.join()

                for step_num, delta_ll, u_index, x_u2 in outputs:
                    self.num_acceptances += 1
                    self.ll += delta_ll
                    self.pts[u_index] = x_u2
                    self.lls.append(self.ll)
                    self.ll_steps.append(step_num)
                pbar.update(self.num_steps - start_steps - pbar.n)

                if type(plot_every) is int and self.num_steps > prev_plot_step + plot_every:
                    self.fix_lls_and_const(prev_plot_step)
                    g_out, A_out, out, percent_edges_captured, percent_fake_edges_wrong = self.get_CM(self.A)
                    self.percent_edges_captureds.append((percent_edges_captured, self.num_steps))
                    self.outs.append((out, self.num_steps))

                    self.plot_ll(n_steps, CM=True, save=save_plot)
                    prev_plot_step = self.num_steps

            self.fix_lls_and_const(prev_plot_step)
            self.plot_ll(n_steps, CM=True)

    def fix_lls_and_const(self, prev_plot_step):
        # quick dirty ll renormalisation
        last_ll_step = -1
        while self.ll_steps[last_ll_step] > prev_plot_step and last_ll_step > -len(self.ll_steps):
            last_ll_step -= 1

        wacky_lls = np.array(self.lls[last_ll_step:])
        last_good_ll = wacky_lls[0]
        self.ll, self.expected_num_edges = self.calculate_ll()
        self.calibrate_const()
        bad_current_ll = self.ll
        better_lls = last_good_ll + ((self.ll - last_good_ll) / (bad_current_ll - last_good_ll)) * (
                    wacky_lls - last_good_ll)
        self.lls[last_ll_step:] = list(better_lls)

    # TODO
    #  This should be faster than run_pool, but for some reason it is a bit slower??
    def run_pool2(self, n_steps, pool_size=5, jobs_per_worker=5, plot_every=None, verbose=False):
        start_steps = self.num_steps
        with tqdm(total=n_steps) as pbar:
            with multiprocessing.Pool(processes=pool_size, initializer=mcmc_girg_init_worker,
                                      initargs=(self.shared_pts, self.pts.shape, self.weights, self.g)) as p:
                while self.num_steps < n_steps + start_steps:
                    seeds = self.num_steps * (jobs_per_worker * pool_size) + np.arange(jobs_per_worker * pool_size)
                    outputs = []
                    results = p.imap_unordered(mcmc_girg_pool_step,
                                               ((seed, self.alpha, self.const_in) for seed in range(n_steps)))
                    for accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2 in results:
                        if verbose:
                            print(f'acceptance_prob: {acceptance_prob}')
                            print(u_index)
                            print(x_u2)
                            print(acceptance_prob)
                            print(f'accepted: {accepted}')


                        # TODO for some reason if you just have this stuff below, the likelihood
                        #  goes down not up. why??
                        # self.num_steps += 1
                        # self.ll += delta_ll
                        # pts = np.frombuffer(self.pts).reshape(self.n, self.d)
                        # pts[u_index] = x_u2
                        # self.num_acceptances += 1
                        # self.lls.append(self.ll)
                        # self.ll_steps.append(self.num_steps)

                        if accepted:
                            outputs.append((self.num_steps, delta_ll, u_index, x_u2))
                        self.num_steps += 1

                        if self.num_steps % (pool_size * jobs_per_worker) == 0:
                            with self.shared_pts.get_lock():
                                pts = np.frombuffer(self.shared_pts.get_obj()).reshape(self.n, self.d)
                                for step_num, delta_ll, u_index, x_u2 in outputs:
                                    self.ll += delta_ll
                                    pts[u_index] = x_u2
                                    self.num_acceptances += 1
                                    self.lls.append(self.ll)
                                    self.ll_steps.append(step_num)

                            outputs = []
                            pbar.update(self.num_steps - start_steps - pbar.n)
                        # pbar.update(self.num_steps - pbar.n)

                        if type(plot_every) is int and self.num_steps % plot_every == 0:
                            self.plot_ll(n_steps)

            p.close()
            p.join()

    def plot_ll(self, n_steps, CM=False, save=False):
        fig = plt.gcf()
        if not CM:
            ax = plt.gca()
        else:
            ax = fig.axes[0]

        ax.clear()
        ax.plot(self.ll_steps, self.lls, 'b')
        ax.set_title(f'n_steps: {self.num_steps} / {n_steps}, num_acceptances: {self.num_acceptances}')
        ax.set_ylabel('log likelihood')

        if CM:
            ax = fig.axes[1]
            ax.clear()
            ax.set_ylabel('percent edges captured in CM')
            ax.plot([x[1] for x in self.percent_edges_captureds], [x[0] for x in self.percent_edges_captureds], 'r')

        if save:
            self.save_plot(fig)
        else:
            fig.canvas.draw()

    def save_plot(self, fig):
        fig_path = os.environ['DATA_PATH'] + 'figures/'
        os.makedirs(fig_path, exist_ok=True)
        fig.savefig(fig_path + self.graph_name + '.png')
    def pickle(self):
        pickle_path = os.environ['DATA_PATH'] + 'pickles/'
        os.makedirs(pickle_path, exist_ok=True)
        fn = pickle_path + self.graph_name + '.pkl'
        with open(fn, 'wb') as file:
            pickle.dump(self, file)

    def acceptances_plot(self, convolve_width=100):
        # running average of mean acceptance rate
        acceptances = np.zeros(self.num_steps+1)
        for step in self.ll_steps:
            acceptances[step] = 1
        plt.plot(np.convolve(acceptances, np.ones(convolve_width) / convolve_width, mode='valid'))

    @staticmethod
    def proposal(k, d, sigma, x_u, p_normal=0.7):
        uniforms = np.random.uniform(size=(k, d))
        normals = np.clip(x_u + np.random.normal(size=(k, d), scale=sigma), 0, 1)
        mask = np.random.binomial(1, p_normal, k)
        return mask * normals + (1 - mask) * uniforms


    def gen_girg(self):
        tau = 2.1 # Ignored
        g, edges, weights, pts, const = generation.generate_GIRG_nk(self.n, self.d, tau, self.alpha, weights=self.weights_original,
                                                                    pts=self.pts,
                                                                    points_type=points.PointsCube)
        return g


    def get_CM(self, A):
        g_out, A_out = self.MC_to_g_A()

        out, percent_edges_captured, percent_fake_edges_wrong = CM(A, A_out)
        return g_out, A_out, out, percent_edges_captured, percent_fake_edges_wrong

    def MC_to_g_A(self):
        tau = 2.1  # Ignored
        g, edges, weights, pts, const = generation.generate_GIRG_nk(
            self.n, self.d, tau, self.alpha, weights=self.weights_original,
            const=self.const,
            pts=self.pts,
            points_type=points.PointsCube)

        gnx = nk.nxadapter.nk2nx(g)
        A = nx.linalg.adjacency_matrix(gnx).todense()
        return g, A

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['shared_pts']
        del state['shared_pts_init']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.pool:
            self.shared_pts = multiprocessing.Array(ctypes.c_double, self.pts.flatten())
            self.pts = self.pts_type(np.frombuffer(self.shared_pts.get_obj()).reshape(self.pts.shape))
            self.shared_pts_init = multiprocessing.Array(ctypes.c_double, self.pts_init.flatten())
            self.pts_init = self.pts_type(np.frombuffer(self.shared_pts_init.get_obj()).reshape(self.pts_init.shape))


# TODO globals are bad, are we using this correctly??
mp_var_dict = {}
fitting_graph = None


def mcmc_girg_init_worker(pts, pts_shape, pts_init, weights, g):
    mp_var_dict['pts'] = pts
    mp_var_dict['pts_shape'] = pts_shape
    # mp_var_dict['pts_init'] = pts_init
    if 'pts_init' in mp_var_dict:
        del(mp_var_dict['pts_init'])

    mp_var_dict['weights'] = weights
    mp_var_dict['g'] = g


def mcmc_girg_pool_step(stuff):
    seed, alpha, const_in, sigma, p_normal = stuff
    np.random.seed(seed=seed)
    # mp_var_dict should be passed in as a setup function
    n, d = mp_var_dict['pts_shape']
    pts = np.frombuffer(mp_var_dict['pts'].get_obj()).reshape(n, d)
    pts = points.PointsCube(pts)

    if 'pts_init' in mp_var_dict:
        pts_init = np.frombuffer(mp_var_dict['pts_init'].get_obj()).reshape(n, d)
    else:
        pts_init = None

    weights = np.frombuffer(mp_var_dict['weights']).reshape(n)
    g = mp_var_dict['g']

    u_index = np.random.randint(n)
    x_u = pts[u_index]
    if pts_init is not None:
        x_u_init = pts_init[u_index]
    else:
        x_u_init = x_u
    # x_u2 = np.random.uniform(size=d)


    # we give this a try. We use both for prior and proposal?
    if sigma is None:
        sigma = 2 * ((1/n)**1/d)
    if p_normal is None:
        p_normal = 0.7

    # trying prior on pts via pts_init
    def prior_x_u(pt):
        out = p_normal * np.sqrt(1/(2*np.pi) * np.exp(-np.linalg.norm(pt - x_u_init)**2 / (2 * sigma**(2*d))))
        + (1 - p_normal)
        return out

    # This is the uniform prior
    def prior_x_u(pt):
        return 1


    # NB! x_u_init is actually the current x_u, since pts_init == None
    x_u2 = MCMC_girg.proposal(1, d, sigma=sigma, x_u=x_u_init, p_normal=p_normal).squeeze()
    acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new = MCMC_girg.acceptance_prob_static(g, weights, alpha, const_in, pts,
                                                                                                  u_index, x_u2, prior_x_u)

    accepted = np.random.rand() < acceptance_prob
    delta_ll = 2 * (u_ll_new - u_ll_old)
    return accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2



def g_diffmap_initialised_mcmc(g, alpha, const, pts_d=1, uniformly=True):

    n = g.numberOfNodes()

    gnx = nk.nxadapter.nk2nx(g)
    A = nx.adjacency_matrix(gnx).todense()

    weights = np.array(utils.graph_degrees_to_weights(g))

    w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.5, eye_or_ones='eye')

    pts_diffmap = np.array([diff_map(i, 10) for i in range(n)])
    if uniformly:
        pts_diffmap = utils.uniformify_pts(pts_diffmap)
    else:
        pts_diffmap = points.normalise_points_to_cube(pts_diffmap)
    pts_diffmap = points.PointsCube(pts_diffmap[:, 0:pts_d])

    pts_init = points.PointsCube(np.random.uniform(size=pts_diffmap.shape))

    # weights = np.array(utils.graph_degrees_to_weights(g))
    MC = MCMC_girg(g, weights, alpha, const, pts_diffmap.copy(), pool=True)
    MC_init = MCMC_girg(g, weights, alpha, const, pts_init.copy())

    return g, A, weights, const, pts_diffmap, pts_init, MC, MC_init

def g_initialised_mcmc(g, alpha, const, pts_d=1, diffmap_init=True, graph_name=None):
    n = g.numberOfNodes()

    gnx = nk.nxadapter.nk2nx(g)
    A = nx.adjacency_matrix(gnx).todense()

    weights = np.array(utils.graph_degrees_to_weights(g))

    if diffmap_init:
        w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.5, eye_or_ones='eye')
        pts = np.array([diff_map(i, 10) for i in range(n)])
        pts = utils.uniformify_pts(pts)
        pts = points.PointsCube(pts[:, 0:pts_d])
    else:
        pts = points.PointsCube(np.random.uniform(size=(n, pts_d)))

    MC = MCMC_girg(g, weights, alpha, const, pts.copy(), pool=True, graph_name=graph_name)

    return g, A, weights, const, pts, MC


def accuracy_experiment(g, d, alpha, const):
    """This will run a diagnostic experiment.
    alpha, const have been estimated already as the best fit for the uniform cube girg.
    We fit the d-dim diffusion map as our initial point estimates
    """
    pass


# Much better
def CM(A, A1):
    """A is the true edge adjacency matrix, A1 is a fake graph.
    Returns a 2x2 matrix of the form
    [[TP, FN],      (first row is where A has edge)
     [FP, TN]]      (second row is where A does not have edge)

     first column is where A1 has edge, second column is where A1 does not have edge
    """
    edge = A != 0

    out = np.array([[np.sum(A1[edge] == 1), np.sum(A1[edge] == 0)],
           [np.sum(A1[~edge] == 1), np.sum(A1[~edge] == 0)]])

    TP, FN, FP, TN = out.flatten()
    percent_edges_captured = TP / (TP + FN)  # of the real edges how many did we capture
    percent_fake_edges_wrong = FP / (FP + TP)  # of the fake edges how many were actually non edges

    return out, percent_edges_captured, percent_fake_edges_wrong

def quick_2dhistplot(g, process='cubify'):
    n = g.numberOfNodes()
    d=2
    w, Phi, Psi, diff_map = utils.get_diffmap(g, Iweighting=0.5, eye_or_ones='eye')
    plt.figure()
    plt.plot(w, marker='o')
    plt.show()

    pts_diffmap = np.array([diff_map(i, 4) for i in range(n)])[:, 0:d]
    if process=='uniformify':
        pts_diffmap = utils.uniformify_pts(pts_diffmap)
    elif process == 'cubify':
        pts_diffmap[:, 1] = cubify_last_k_dim(pts_diffmap, k=1, num_near=50)
        pts_diffmap[:, 0] = pts_diffmap[:, 0].argsort().argsort()/n
    pts_diffmap = points.PointsCube(pts_diffmap)


    # a = MC.pts[:, 0]
    # plt.hist(a[a<0.04], bins=50)

    plt.figure()
    plt.scatter(pts_diffmap[:, 0], pts_diffmap[:, 1])
    plt.show()
    plt.figure()
    plt.hist2d(pts_diffmap[:, 0], pts_diffmap[:, 1], bins=30)
    plt.figure()
    sns.displot(x=pts_diffmap[:, 0], y=pts_diffmap[:, 1], kind="kde")
    return pts_diffmap

