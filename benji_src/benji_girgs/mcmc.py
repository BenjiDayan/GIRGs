import networkit as nk
import numpy as np
from benji_girgs import points, generation
import multiprocessing
import matplotlib.pyplot as plt
import ctypes
from tqdm import tqdm


class MCMC_girg():
    def __init__(self, g: nk.Graph, weights: np.ndarray, alpha: float, const: float, pts: points.PointsCube,
                 pool=False):
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
        else:
            self.pts = pts

        # Will this work??
        global fitting_graph
        fitting_graph = g
        self.g = g

        self.weights = weights / np.sqrt(weights.sum())
        self.const_in = generation.const_conversion(const, alpha, d=1, true_volume=True)
        self.alpha = alpha
        # self.p_uv = generation.get_probs(weights / np.sqrt(weights.sum()), self.pts, alpha, self.const_in)
        # self.u_lls = []
        self.ll = 0  # This is going to be double the actual LL as we double count edges
        for u_index in range(self.n):
            eps = 1e-7
            p_u_to_vs = generation.get_probs_u(self.weights, self.pts, self.alpha, self.const_in, u_index)
            p_u_to_vs = np.clip(p_u_to_vs, eps, 1 - eps)
            u_ll = self.p_u_to_vs_to_ll(self.g, u_index, p_u_to_vs)
            self.ll += u_ll

        self.ll_steps = [0]
        self.lls = [self.ll]
        self.num_acceptances = 0
        self.num_steps = 0

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
    def acceptance_prob_static(g, weights, alpha, const_in, pts, u_index, x_u2):
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

        Q_ratio = np.exp(u_ll_new - u_ll_old)
        acceptance_prob = min(1, Q_ratio)
        return acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs, p_u_to_vs2

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
        seeds = self.num_steps * (jobs_per_worker * pool_size) + np.arange(jobs_per_worker * pool_size)
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
        with tqdm(total=n_steps) as pbar:
            while self.num_steps < n_steps:
                if pool_size:
                    self.step_pool(self.num_steps, pool_size, jobs_per_worker)
                else:
                    self.step2()
                    self.num_steps += 1

                if type(plot_every) is int and self.num_steps % plot_every == 0:
                    self.plot_ll(n_steps)

                pbar.update(self.num_steps - pbar.n)

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

    def run_pool(self, n_steps, pool_size=5, jobs_per_worker=5, plot_every=None, verbose=False):
        with tqdm(total=n_steps) as pbar:
            while self.num_steps < n_steps:
                seeds = self.num_steps*(jobs_per_worker*pool_size) + np.arange(jobs_per_worker*pool_size)
                outputs = []
                with multiprocessing.Pool(processes=pool_size, initializer=mcmc_girg_init_worker, initargs=(self.shared_pts, self.pts.shape, self.weights, self.g)) as p:
                    results = p.imap_unordered(mcmc_girg_pool_step, [(seed, self.alpha, self.const_in) for seed in seeds], chunksize=jobs_per_worker)
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
                    self.num_steps += 1
                    self.ll += delta_ll
                    self.pts[u_index] = x_u2
                    self.num_acceptances += 1
                    self.lls.append(self.ll)
                    self.ll_steps.append(self.num_steps)
                pbar.update(self.num_steps - pbar.n)

                if type(plot_every) is int and self.num_steps % plot_every == 0:
                    self.plot_ll(n_steps)

    # This should be faster than run_pool, but for some reason it isn't really.
    def run_pool2(self, n_steps, pool_size=5, jobs_per_worker=5, plot_every=None, verbose=False):
        with tqdm(total=n_steps) as pbar:
            with multiprocessing.Pool(processes=pool_size, initializer=mcmc_girg_init_worker,
                                      initargs=(self.shared_pts, self.pts.shape, self.weights, self.g)) as p:
                while self.num_steps < n_steps:
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
                            outputs.append((delta_ll, u_index, x_u2))
                        self.num_steps += 1

                        if self.num_steps % (pool_size * jobs_per_worker) == 0:
                            with self.shared_pts.get_lock():
                                pts = np.frombuffer(self.shared_pts.get_obj()).reshape(self.n, self.d)
                                for delta_ll, u_index, x_u2 in outputs:
                                    self.num_steps += 1
                                    self.ll += delta_ll
                                    pts[u_index] = x_u2
                                    self.num_acceptances += 1
                                    self.lls.append(self.ll)
                                    self.ll_steps.append(self.num_steps)

                            outputs = []
                            pbar.update(self.num_steps - pbar.n)
                        # pbar.update(self.num_steps - pbar.n)

                        if type(plot_every) is int and self.num_steps % plot_every == 0:
                            self.plot_ll(n_steps)

            p.close()
            p.join()

    def plot_ll(self, n_steps):
        ax = plt.gca()
        fig = plt.gcf()
        ax.clear()
        ax.plot(self.ll_steps, self.lls)
        ax.set_title(f'n_steps: {self.num_steps} / {n_steps}, num_acceptances: {self.num_acceptances}')
        fig.canvas.draw()

    def acceptances_plot(self):
        # running average of mean acceptance rate
        acceptances = np.zeros(self.num_steps)
        for step in self.ll_steps:
            acceptances[step] = 1
        plt.plot(np.convolve(acceptances, np.ones(100) / 100, mode='valid'))


# TODO globals are bad, are we using this correctly??
mp_var_dict = {}
fitting_graph = None


def mcmc_girg_init_worker(pts, pts_shape, weights, g):
    mp_var_dict['pts'] = pts
    mp_var_dict['pts_shape'] = pts_shape
    mp_var_dict['weights'] = weights
    mp_var_dict['g'] = g


def mcmc_girg_pool_step(stuff):
    seed, alpha, const_in = stuff
    np.random.seed(seed=seed)
    # mp_var_dict should be passed in as a setup function
    n, d = mp_var_dict['pts_shape']
    pts = np.frombuffer(mp_var_dict['pts'].get_obj()).reshape(n, d)
    pts = points.PointsCube(pts)
    weights = np.frombuffer(mp_var_dict['weights']).reshape(n)
    g = mp_var_dict['g']

    u_index = np.random.randint(n)
    x_u2 = np.random.uniform(size=d)
    acceptance_prob, u_ll_old, u_ll_new, p_u_to_vs_old, p_u_to_vs_new = MCMC_girg.acceptance_prob_static(g, weights, alpha, const_in, pts,
                                                                                                  u_index, x_u2)

    accepted = np.random.rand() < acceptance_prob
    delta_ll = 2 * (u_ll_new - u_ll_old)
    return accepted, acceptance_prob, delta_ll, u_index, u_index, x_u2

