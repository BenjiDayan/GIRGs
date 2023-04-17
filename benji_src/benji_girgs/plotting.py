import networkit as nk
import matplotlib.pyplot as plt
import numpy as np
from networkit.graph import Graph 

import powerlaw


def plot_degree_dist(g: Graph, pl_fit=False, vlines=0):
    if type(g) is nk.Graph:
        dd = sorted(nk.centrality.DegreeCentrality(g).run().scores(), reverse=True)
    elif type(g) is np.ndarray and np.issubdtype(g.dtype, np.integer):
        dd = sorted(g.astype(np.int64), reverse=True)
    else:
        raise Exception('g should be an nk Graph, or a np.ndarray of integers >=1')
    degrees, numberOfNodes = np.unique(dd, return_counts=True)
    # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplot(121)
    # plt.sca(axes[0])
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("number of nodes")
    # plt.scatter(degrees, numberOfNodes, s=1.1, marker='x')
    plt.scatter(degrees, numberOfNodes)
    if pl_fit:
        fit = powerlaw.Fit(dd, discrete=True)
        print(f'powerlaw alpha: {fit.power_law.alpha:.3f}')
#         fit.power_law.plot_pdf(linestyle='--', color='purple')
        plt.axvline(fit.xmin, linestyle='--', color='r', label=f'xmin: {fit.xmin}')
#         plt.axvline(fit.xmax, linestyle='--', color='pink', label=f'xmax: {fit.xmax}')
        y = fit.power_law.pdf()
        plt.plot(fit.data, y * len(fit.data), linestyle='--', color='purple')
        
    if vlines > 0:  # plot like quartile lines for number of nodes.
        # rough q-tiles
        q = vlines
        colors = plt.cm.rainbow(np.linspace(0, 1, q))
        rev_dd = list(reversed(dd))
        for i in range(1, q):
            plt.axvline(rev_dd[i * len(dd)//q], label=f'qtile-{i}/{q}', c=colors[i])
        plt.legend()
    # plt.show()
    plt.subplot(122)

    one_minus_cdf = 1. * np.arange(len(dd)) / (len(dd) - 1)
    plt.xscale("log")
    plt.xlabel("degree")
    plt.yscale("log")
    plt.ylabel("1 - CDF")
    plt.plot(dd, one_minus_cdf)
    ax = plt.gca()
    if pl_fit:
        y = fit.power_law.ccdf()
        perc = len(fit.data)/len(fit.data_original)
#         fit.power_law.plot_ccdf(linestyle='--', color='purple', ax=ax)
        plt.plot(fit.data, y * perc, linestyle='--', color='purple')
        plt.axvline(fit.xmin, linestyle='--', color='r', label=f'xmin: {fit.xmin}')
#         plt.axvline(fit.xmax, linestyle='--', color='pink', label=f'xmax: {fit.xmax}')


    if vlines > 0:  # plot like quartile lines for number of nodes.
        # rough q-tiles
        q = vlines
        colors = plt.cm.rainbow(np.linspace(0, 1, q))
        rev_dd = list(reversed(dd))
        for i in range(1, q):
            plt.axvline(rev_dd[i * len(dd)//q], label=f'qtile-{i}/{q}', c=colors[i])
        plt.legend()



def reg_std_gdist_alphas_plot(n, d, tau, alphas=[1.02, 1.08, 1.3, 1.6, 2.0], target_degree=20.0):
    """
    Plot the regularised standard graph distance as a function of alpha
    """
    df = pd.DataFrame(columns=['alpha', 'std_gdist'])
    i=0
    for alpha in alphas:
        print(alpha)
        for _ in range(8):
            g, edges, weights, pts, c, id2gnk = cgirg_gen(n, d, tau, alpha, desiredAvgDegree=target_degree, weights=None)
            try:
                std_gdist = regularised_std_graph_distance(g)
                df.loc[i] = [alpha, std_gdist]
            except ValueError:
                pass
            i += 1
        
        
    # df['mean_std_gdist'] = df.std_gdists.apply(lambda x: np.mean(x))
    # df['std_std_gdist'] = df.std_gdists.apply(lambda x: np.std(x))

    df['alpha2'] = df.alpha.apply(lambda x: str(x))

    sns.swarmplot(data=df, x='alpha2', y='std_gdist')
    return df

def metric_func_alphas_plot(n, d, tau, target_degree, metric_func, metric_func_name, alphas=[1.02, 1.1, 1.3, 1.6, 2.0]):
    df = pd.DataFrame(columns=['alpha', 'metric_func_name'])
    i=0
    for alpha in alphas:
        print(alpha)
        for _ in range(8):
            g, edges, weights, pts, c, id2gnk = cgirg_gen(n, d, tau, alpha, desiredAvgDegree=target_degree, weights=None)
            try:
                metric = metric_func(g)
                df.loc[i] = [alpha, metric]
            except Exception:
                pass
            i += 1
        
        
    # df['mean_std_gdist'] = df.std_gdists.apply(lambda x: np.mean(x))
    # df['std_std_gdist'] = df.std_gdists.apply(lambda x: np.std(x))

    df['alpha2'] = df.alpha.apply(lambda x: str(x))

    sns.swarmplot(data=df, x='alpha2', y=metric_func_name)
    return df