import pytest
import networkit as nk
from benji_girgs import utils
import numpy as np

@pytest.fixture
def small_graph():
    g = nk.Graph(5)
    g.addEdge(1, 2)
    g.addEdge(2, 3)
    g.addEdge(3, 1)
    g.addEdge(1, 4)

    return g

def test_avg_degree(small_graph):
    assert utils.avg_degree(small_graph) == 1.6

def test_LCC(gsmall_graph):
    assert utils.LCC(small_graph) == np.mean([0, 1/3, 1, 1, 0])


def test_get_largest_component(small_graph):
    g_out = utils.get_largest_component(small_graph)
    assert set(g_out.iterNodes()) == {0, 1, 2, 3}
    assert set(g_out.iterEdges()) == {(0, 1), (1, 2), (0, 2), (0, 3)}
