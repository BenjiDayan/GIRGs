import pytest
from benji_girgs import generation
import networkit as nk


if __name__ == '__main__':
    n, d, tau, alpha = 1000, 2, 2.5, 1.3
    gnk, edges, weights, pts, const = generation.cgirg_gen(n, d, tau, alpha, desiredAvgDegree=20.0)
    nk.overview(gnk)