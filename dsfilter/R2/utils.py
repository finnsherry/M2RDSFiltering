"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used on R^2.
"""

import numpy as np
import taichi as ti
from dsfilter.utils import linear_interpolate

# Safe Indexing

@ti.func
def sanitize_index(
    index: ti.types.vector(2, ti.i32),
    input: ti.template()
) -> ti.types.vector(2, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`. Adapted from Gijs.

    Args:
        `index`: ti.types.vector(n=2, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=2, dtype=ti.i32) of index that is within `input`.
    """
    shape = ti.Vector(ti.static(input.shape), dt=ti.i32)
    return ti.Vector([
        ti.math.clamp(index[0], 0, shape[0] - 1),
        ti.math.clamp(index[1], 0, shape[1] - 1),
    ], dt=ti.i32)

@ti.func
def sanitize_reflected_index(
    index: ti.types.vector(2, ti.i32),
    input: ti.template()
) -> ti.types.vector(2, ti.i32):
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`, while reflecting at
    the boundaries.

    Args:
        `index`: ti.types.vector(n=2, dtype=ti.i32) index.
        `input`: ti.field in which we want to index.

    Returns:
        ti.types.vector(n=2, dtype=ti.i32) of index that is within `input`.
    """
    I, J = index
    I_max, J_max = ti.Vector(ti.static(input.shape), dt=ti.i32) - ti.Vector([1, 1], dt=ti.i32)
    return ti.Vector([
        -I * (I < 0) + I * (0 <= I <= I_max) + (2 * I_max - I) * (I > I_max),
        -J * (J < 0) + J * (0 <= J <= J_max) + (2 * J_max - J) * (J > J_max),
    ], dt=ti.i32)

@ti.func
def index_reflected(
    input: ti.template(),
    index: ti.types.vector(2, ti.i32)
) -> ti.f32:
    """
    @taichi.func
    
    Make sure the `index` is inside the shape of `input`, while reflecting at
    the boundaries.

    Args:
        `input`: ti.field in which we want to index.
        `index`: ti.types.vector(n=2, dtype=ti.i32) index.

    Returns:
        Value of `input` at `index`, taking care of reflected boundaries.
    """
    index = sanitize_reflected_index(index, input)
    return input[index]

# @ti.kernel
# def clamp_field(
#     u: ti.template(),
#     clip_l: ti.f32,
#     clip_r: ti.f32,
# ):
#     for I in ti.grouped(u):
#         u[I] = ti.math.clamp(u[I], clip_l, clip_r)

# Interpolate

@ti.func
def bilinear_interpolate(
    v00: ti.f32,
    v01: ti.f32,
    v10: ti.f32,
    v11: ti.f32,
    r: ti.types.vector(2, ti.f32)
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v**` depending on the distance `r`, via 
    repeated linear interpolation (x, y). Adapted from Gijs.

    Args:
        `v**`: values at points between which we want to interpolate, taking 
          real values.
        `r`: ti.types.vector(n=2, dtype=[float]) defining the distance to the
          points between which we to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
        Interpolated value.
    """
    v0 = linear_interpolate(v00, v10, r[0])
    v1 = linear_interpolate(v01, v11, r[0])

    v = linear_interpolate(v0, v1, r[1])

    return v

@ti.func
def scalar_bilinear_interpolate(
    input: ti.template(),
    index: ti.template()
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of `input` at continuous `index` bilinearly, via repeated
    linear interpolation (x, y). Adapted from Gijs.

    Args:
        `input`: ti.field(dtype=[float]) in which we want to interpolate.
        `index`: ti.types.vector(n=2, dtype=[float]) continuous index at which
          we want to interpolate.

    Returns:
        Value of `input` interpolated at `index`.
    """
    r = ti.math.fract(index)

    f = ti.math.floor(index, ti.i32)
    f = sanitize_index(f, input)

    c = ti.math.ceil(index, ti.i32)
    c = sanitize_index(c, input)

    v00 = input[f[0], f[1]]
    v01 = input[f[0], c[1]]
    v10 = input[c[0], f[1]]
    v11 = input[c[0], c[1]]

    v = bilinear_interpolate(v00, v01, v10, v11, r)

    return v

# Coordinate Transforms

def coordinate_real_to_array(x, y, x_min, y_min, dxy):
    """
    Compute the array indices (I, J) of the point defined by real coordinates 
    (`x`, `y`). Can broadcast over entire arrays of real coordinates.

    Args:
        `x`: x-coordinate of the point.
        `y`: y-coordinate of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    I = np.rint((x - x_min) / dxy).astype(int)
    J = np.rint((y - y_min) / dxy).astype(int)
    return I, J


def coordinate_array_to_real(I, J, x_min, y_min, dxy):
    """
    Compute the real coordinates (x, y) of the point defined by array indices 
    (`I`, `J`). Can broadcast over entire arrays of array indices.

    Args:
        `I`: I index of the point.
        `J`: J index of the point.
        `x_min`: minimum value of x-coordinates in rectangular domain.
        `y_min`: minimum value of y-coordinates in rectangular domain.
        `dxy`: spatial resolution, which is equal in the x- and y-directions,
          taking values greater than 0.
    """
    x = x_min + I * dxy
    y = y_min + J * dxy
    return x, y