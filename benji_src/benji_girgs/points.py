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

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj
    def dists(self) -> np.ndarray:
        """
        Returns an (n,n) matrix of distances between points
        """
        pass

    def dist(self, other: "Points") -> "Points":
        """
        self, other should both be same (n, d) shape. So now we return pairwise distances,
        a (n,) vector
        """
        return get_dist(self, other)

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
        return get_dists(self.astype(np.float16))

    def dist(self, other):
        return get_dist(self.astype(np.float16), other.astype(np.float16))

# Potentiallly we get rid of this, but its a "True to volume" version of PointsTorus - PointsTorus matches
# C++ GIRGs.
class PointsTorus2(PointsTrue):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self, b_vec=None):
        return 2*get_dists(self.astype(np.float16), b_vec=b_vec)

    def dist(self, other, b_vec=None):
        return 2*get_dist(self.astype(np.float16), other.astype(np.float16), b_vec=b_vec)




# This class is useful as a cube GIRG indicator, although we never atcually use the
# dists function directly, rather cgirg_gen_cube_coupling uses base Torus Points.
class PointsCube(PointsTrue):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    def dists(self, b_vec=None):
        return 2*get_dists_cube(self.astype(np.float16), b_vec=b_vec)

    def dist(self, other, b_vec=None):
        return 2*get_dist_cube(self.astype(np.float16), other.astype(np.float16), b_vec=b_vec)

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

def get_points_distorted(b_vec, cube=False):
    multiplier = np.prod(b_vec)
    multiplier = (1/multiplier)**(1/len(b_vec))
    class PointsDistorted(PointsTorus2 if not cube else PointsCube):
        """We put ratios on the distances. I.e. [0, 1]^d Torus, but now
        have b_1, ..., b_d >= 0 which are the scale factors.
        r(x, y) = Max b_i |x_i - y_i|

        Volume(Torus=[0,1])^d = Product_i b_i
        Volume(Ball of radius r) = Product_i (2r)/b_i = r^d Volume(Ball of radius 1)

        So [Vol(B_r) / Vol(T)]^(1/d) is what we want to output.
        = [ (2r)^d * Product_i (1/b_i^2) ]^(1/d)

        I'm actually not certain but I think the multiplier
        we want is actually
        [ Product_i (1/b_i) ]^(1/d).

        This is because I guess volumes are still true volumes,
        Vol(T) = 1, but the distance r is scaled.
        Maybe the error correction for c has to be changed?

        It seems like this whole system is not quite perfect. Maybe because reasons?


        """

        def dists(self):
            n, d = self.shape
            assert len(b_vec) == d
            r = super().dists(b_vec=b_vec)
            # r = get_dists(self.astype(np.float16), max=True, b_vec=b_vec)
            return r * multiplier

        def dist(self, other):
            n, d = self.shape
            assert (n, d == other.shape)
            assert len(b_vec) == d
            r = super().dist(other, b_vec=b_vec)
            return r * multiplier

    return PointsDistorted


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
def get_dists(torus_points: np.ndarray, max=True, b_vec=None):
    diff = np.abs(torus_points[:, None, :] - torus_points[None, :, :])
    torus_diff = np.minimum(diff, 1 - diff)
    if b_vec is not None:
        # should be (d,) and (n, d)
        assert b_vec.shape[0] == torus_points.shape[1]
        torus_diff *= b_vec
    # dists = np.linalg.norm(torus_diff, ord=np.inf, axis=-1)
    dists = torus_diff.max(axis=-1) if max else torus_diff.min(axis=-1)
    return dists

def get_dist(torus_points1: np.ndarray, torus_points2: np.ndarray, max=True, b_vec=None):
    """(n, d) x (n, d) -> (n,) pairwise distances"""
    diff = np.abs(torus_points1 - torus_points2)
    torus_diff = np.minimum(diff, 1 - diff)
    if b_vec is not None:
        # should be (d,) and (n, d)
        assert b_vec.shape[0] == torus_points1.shape[1]
        torus_diff *= b_vec
    dists = torus_diff.max(axis=-1) if max else torus_diff.min(axis=-1)
    return dists

# TODO reinstate julia version
# Fastest is julia version of Minimal extra memory
def get_dists_julia(torus_points: np.ndarray):
    torus_points = torus_points.astype(np.float16)
    # return jl.get_dists_novars(torus_points)
    return get_dists(torus_points, max=True)


def get_dists_cube(points: np.ndarray, b_vec=None):
    if b_vec is not None:
        # should be (d,) and (n, d)
        assert b_vec.shape[0] == points.shape[1]
        points *= b_vec

    return np.linalg.norm(points[:, None, :] - points[None, :, :], ord=np.inf, axis=-1)

def get_dist_cube(points1: np.ndarray, points2: np.ndarray, b_vec=None):
    if b_vec is not None:
        # should be (d,) and (n, d)
        assert b_vec.shape[0] == points1.shape[1]
        points1 *= b_vec
        points2 *= b_vec
    return np.linalg.norm(points1 - points2, ord=np.inf, axis=-1)

def get_dists_mcd(points: np.ndarray):
    # return np.min(np.abs(points[:, None, :] - points[None, :, :]), axis=-1)
    points = points.astype(np.float16)
    # return jl.get_dists_novars_min(points)
    return get_dists(points, max=False)



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


def normalise_points_to_cube(pts):
    """If pts is nxd shape, then we will make them all fit to the [0,1]^d cube"""
    pts_out = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))
    return pts_out