# GIRGs

Thesis pdf: [thesis/thesis.pdf](thesis/thesis.pdf)

Framework for real graph and generative graph model comparison: [nemo-eva](nemo-eva)
- See: Bläsius, Thomas, et al. "Towards a systematic evaluation of generative network models." Algorithms and Models for the Web Graph: 15th International Workshop, WAW 2018, Moscow, Russia, May 17-18, 2018, Proceedings 15. Springer International Publishing, 2018.

[Jupyter-on-Euler-or-Leonhard-Open](Jupyter-on-Euler-or-Leonhard-Open) - adapted from https://gitlab.ethz.ch/sfux/Jupyter-on-Euler-or-Leonhard-Open, for running a jupyter server on Euler


## Project Structure

    [benji_src](benji_src)
    ├── do_*.py                 # scripts / experiments, e.g. do_MCMC_ordered.py fits GIRG node locations for a real graph by iterative improvement
    ├── notebooks/*.ipynb       # many messy jupyter notebooks for testing code and some data analysis
    ├── tests/*.py              # dearth of tests
      ...
    └── benji_girgs             # Main python backbone for functions and tools
        ├── fitting.py              # DEPRECATED attempts to fit a GIRG to a real graph
        ├── generation.py           # generating GIRGs (cube/torus, max norm / MCD etc.).
                                      Also Chung-Lu generation/fitting, power law distributions.
                                      generate_GIRG_nk is the main omni-function for generating a GIRG
        ├── graph_kernels.py        # testing out graph kernels to distinguish graphs
        ├── mcmc.py                 # iteratively refining GIRG fit point locations: MCMC / likelihood maximisation
        ├── plotting.py             # plotting graph node degree distributions with their power law fit tails
        ├── points.py               # classes for different distance functions: Max, MCD, Boolean mixed, distorted
        └── utils.py                # Utils for working with graphs. Including diffusion maps for initial node location estimates


Main framework experimental results run in [benji_src/do_feature_extract.py](benji_src/do_feature_extract.py): `os.environ[DATA_PATH]` is location of the folder containing `socfb-*` graphs. Fitting different generative models to real graphs, generating a mirrored synthetic graph dataset, and recording features of all real and generated graphs is done by `FeatureExtractor` class in [nemo-eva/src/feature_extractor.py](nemo-eva/src/feature_extractor.py). We have modified this class to additionally include GIRG fitting and generation.

The extracted features are then cleaned in [nemo-eva/src/feature_cleaner.py](nemo-eva/src/feature_cleaner.py), and a real/synthetic classifier trained and evaluated on each feature set and generative model combination in [nemo-eva/src/main.py](nemo-eva/src/main.py).

The experiments were run on the ETH Zurich Euler cluster, which has `Intel(R) Xeon(R) CPU E3-1284L v4 @ 2.90GHz` cpus, generally requesting 10-12 CPUs and 16-24 GB of RAM, and taking a couple of hours to a day or two to complete (FeatureExtraction - cleaning and classification is much faster).