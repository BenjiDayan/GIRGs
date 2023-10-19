# GIRGs
My (Benjamin Dayan's) master's thesis at ETH, under the supervision of Marc Kaufmann, Ulysse Schaller, Prof. Dr. Johannes Lengler and Prof. Dr. Angelika Steger

[thesis.pdf](thesis/thesis.pdf)

## Project Structure

    benji_src
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

    thesis/						# thesis report latex
    nemo-eva/					# Modified Bläsius et al. framework
    girg-sampling/				# Python wrapper of Bläsius et al. C++ GIRG generation code

### Network Model Evaluation Framework (NeMo-Eva)
The first half of the thesis is extending the framework of Bläsius et al. to include a range of GIRG variants. Modified framework source code is in [nemo-eva](nemo-eva)

> Thomas Bläsius, Tobias Friedrich, Maximilian Katzmann, Anton Krohmer and Jonathan Striebel  
[Towards a Systematic Evaluation of Generative Network Models](https://hpi.de/friedrich/news/2018/waw.html?tx_extbibsonomycsl_publicationlist%5BuserName%5D=puma-friedrich&tx_extbibsonomycsl_publicationlist%5BintraHash%5D=ba31d2c6fa65ad94fee206e6d3ec477c&tx_extbibsonomycsl_publicationlist%5BfileName%5D=TowardsASystematicEvaluationOfGenerativeNetworkModels.pdf&tx_extbibsonomycsl_publicationlist%5Baction%5D=download&tx_extbibsonomycsl_publicationlist%5Bcontroller%5D=Document&cHash=aebce46aee1869cfc63db19e5e7f63f9)  
[15th Workshop on Algorithms and Models for the Web Graph (2018)](http://www.math.ryerson.ca/waw2018/)  
[Part of the Lecture Notes in Computer Science book series (LNCS, volume 10836)](https://link.springer.com/chapter/10.1007/978-3-319-92871-5_8)


We also make use of a Bläsius linear time GIRG generating code in C++, in combination with our own derived algorithms for sampling a cube GIRG and Boolean / MCD GIRGs
> Bläsius, T., Friedrich, T., Katzmann, M., Meyer, U., Penschuck, M., & Weyand, C. [Efficiently generating geometric inhomogeneous and hyperbolic random graphs](https://www.cambridge.org/core/journals/network-science/article/efficiently-generating-geometric-inhomogeneous-and-hyperbolic-random-graphs/EE2080A5FEC3A6C2B3AB451934A340AC)


### Experiments

`os.environ[DATA_PATH]` is location of the folder containing `socfb-*` graphs

#### NeMo-Eva experiments
Main framework experimental results run in [benji_src/do_feature_extract.py](benji_src/do_feature_extract.py): Fitting different generative models to real graphs, generating a mirrored synthetic graph dataset, and recording features of all real and generated graphs is done by `FeatureExtractor` class in [nemo-eva/src/feature_extractor.py](nemo-eva/src/feature_extractor.py). We have modified this class to additionally include GIRG fitting and generation.

The extracted features are then cleaned in [nemo-eva/src/feature_cleaner.py](nemo-eva/src/feature_cleaner.py), and a real/synthetic classifier trained and evaluated on each feature set and generative model combination in [nemo-eva/src/main.py](nemo-eva/src/main.py).

The experiments were run on the ETH Zurich Euler cluster, which has `Intel(R) Xeon(R) CPU E3-1284L v4 @ 2.90GHz` cpus, generally requesting 10-12 CPUs and 16-24 GB of RAM, and taking a couple of hours to a day or two to complete (FeatureExtraction - cleaning and classification is much faster).

#### GIRG node location fitting experiments
Main point fitting experiments run in [benji_src/do_MCMC_ordered.py](benji_src/do_MCMC_ordered.py): Using a diffusion map initial estimate of GIRG node locations, and then iteratively improving locations by ordered likelihood maximisation.

The final fit point locations are saved, for later edge accuracy metric analysis.

