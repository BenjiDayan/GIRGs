import numpy as np
from typing import List

import os

from julia.api import Julia
jl_temp = Julia(compiled_modules=False)
from julia import Main as jl
file_path = os.path.dirname(os.path.abspath(__file__)) + '/benji_jl_dists.jl'
jl.file_path = file_path
jl.eval('include(file_path)')


class Points(np.ndarray):
    """
    Base Class for (n, d) array of points.
    """
    def dists(self) -> np.ndarray:
        """
        Returns an (n,n) matrix of distances between points
        """
        pass

# True to Volume version of Points - need adjustments in const
class PointsTrue(Points):
    pass

class PointsTorus(Points):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self):
        return get_dists_julia(self)

# Potentiallly we get rid of this, but its a "True to volume" version of PointsTorus - PointsTorus matches
# C++ GIRGs.
class PointsTorus2(PointsTrue):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self):
        n, d = self.shape
        return 2*get_dists_julia(self)


    
class PointsCube(Points):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self):
        return get_dists_cube(self)

########## NB
# For PointsMCD and PointsSimpleMixed, we actually exponentiate the distances
# by 1/d. This is because in get_probs, each distance r_uv is raised to the power
# of d. So we need to undo that here, since V_min(r_uv) := r_uv. (This is actually
# a simplification, it's actually 2r <= 1 - (1 - 2r)^d <= 2dr in the paper, which
# they then simplify to just r (as we do here).)
#
# note that the formula P(A u B u C) = 1 - (1-2r)^3 for d=3 e.g. comes from
# P(A u B u C) = P( (A^c n B^c n C^c)^c ) = 1 - P(A^c n B^c n C^c)
# = 1 - P(A^c)P(B^c)P(C^c) = 1 - (1-2r)^3
#
# However our codebase throughout actualy samples from the n**(1/d) side length torus.
# Hence we'd need to replace 1 - (1-2r)^d => n - (n^(1/d) - 2r)^d ~= n^[(d-1)/d] r I guess?
# This kind of makes sense:
#   previously r^d --(r=n^(1/d))-> n = Vol(Torus)
#   now, n^[(d-1)/d] r --(r=n^(1/d))-> n^[(d-1)/d] n^(1/d) = n
#
# We could also derive this from the formula from one section: (2r) * (n^(1/d))^(d-1).
# The intersection of d of these planes has volume basically (2dr) * (n^(1/d))^(d-1),
# which we then scale down to just r * (n^(1/d))^(d-1) = n^[(d-1)/d] r.
#
# This gives us the volume for a Min Max mix:
# So if we group the d dimensions into a1, a2, ..., ak disjoint groups, then the volume
# Vol(r) = n - Prod_{i=1}^k (n^(|a_i|/d) - (2r)^|a_i|) =
#
# I think we'd best use this full formula throughout for consistency
class PointsMCD(PointsTrue):

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def dists(self):
        n, d = self.shape
        r = get_dists_mcd(self)
        # return (n**((d-1)/d) * r)**(1/d)
        # out = n - (n**(1/d) - 2*r)**d
        out = 1 - (1 - 2*r)**d
        # out = 1 - (1 - r)**d
        return out**(1/d)
def get_points_simple_mixed_class(groups):
    """
    E.g. if groups = [[0], [1,2]]
    We get a Min(diff0, Max(diff1, diff2)) distance for absolue torus distances.

    """
    class PointsSimpleMixed(PointsTrue):
        my_groups = groups
        def __new__(cls, input_array):
            obj = np.asarray(input_array).view(cls)
            return obj

        def dists(self):
            n, d = self.shape
            r = get_dists_mixed(self, self.my_groups)
            prod = 1
            for group in self.my_groups:
                prod *= (1 - (2*r)**len(group))
            out = 1 - prod
            return out**(1/d)

    return PointsSimpleMixed


# class PointsSimpleMixed(Points):
#     def __new__(cls, input_array):
#         obj = np.asarray(input_array).view(cls)
#         return obj
#
#     def dists(self):
#         return get_dists_simple_mixed(self)


##### Deprecated, and use torus_side_length

# lots of extra memory
def get_dists2(torus_points: np.ndarray, torus_side_length):
    """
    torus_points: (n x d) array of points
    """
    n = len(torus_points)
    lpts = np.tile(torus_points, (n, 1))
    rpts = np.repeat(torus_points, n, axis=0)
    diff = np.abs(rpts - lpts)
    torus_diff = np.minimum(diff, torus_side_length - diff)
    dists = np.linalg.norm(torus_diff, ord=np.inf, axis=1)
    return dists.reshape(n, n)

##### End Deprecated


# Minimal extra memory
# This one seems hella faster
def get_dists(torus_points: np.ndarray):
    diff = np.abs(torus_points[:, None, :] - torus_points[None, :, :])
    torus_diff = np.minimum(diff, 1 - diff)
    dists = np.linalg.norm(torus_diff, ord=np.inf, axis=-1)
    return dists


# Fastest is julia version of Minimal extra memory
def get_dists_julia(torus_points: np.ndarray):
    torus_points = torus_points.astype(np.float16)
    return jl.get_dists_novars(torus_points)


def get_dists_cube(points: np.ndarray):
    return np.linalg.norm(points[:, None, :] - points[None, :, :], ord=np.inf, axis=-1)


def get_dists_mcd(points: np.ndarray):
    # return np.min(np.abs(points[:, None, :] - points[None, :, :]), axis=-1)
    points = points.astype(np.float16)
    return jl.get_dists_novars_min(points)



def get_dists_mixed(points: np.ndarray, groups: List[List[int]]):
    """E.g. groups = [[0], [1, 2], [0, 3], ...] means take
    Min(
        Max(|z_0 - z_0'|),
        Max(|z_1 - z_1'|, |z_2 - z_2'|),
        Max(|z_0 - z_0'|, |z_3 - z_3'|),
        ...
    )"""
    out = get_dists_julia(points[:, groups[0]])
    for group in groups[1:]:
        out = np.minimum(out, get_dists_julia(points[:, group]))
    return out