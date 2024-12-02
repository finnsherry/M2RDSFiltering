"""
    utils
    =====

    Provides miscellaneous computational utilities that can be used both on R^2
    and M_2.
"""

import numpy as np
import taichi as ti


# Interpolation

@ti.func
def linear_interpolate(
    v0: ti.f32,
    v1: ti.f32,
    r: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Interpolate value of the points `v*` depending on the distance `r`, via 
    linear interpolation. Adapted from Gijs.

    Args:
        `v*`: values at points between which we want to interpolate, taking real 
          values.
        `r`: distance to the points between which we to interpolate, taking real
          values.

    Returns:
        Interpolated value.
    """
    return v0 * (1.0 - r) + v1 * r

# Derivatives

@ti.func
def select_upwind_derivative_dilation(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Select the correct derivative for the upwind derivative of dilation.

    Args:
        `d_forward`: derivative in the forward direction.
        `d_backward`: derivative in the backward direction.
          
    Returns:
        derivative in the correct direction.
    """
    return ti.math.max(d_forward, -d_backward, 0) * (-1.)**(d_forward <= -d_backward)

@ti.func
def select_upwind_derivative_erosion(
    d_forward: ti.f32,
    d_backward: ti.f32
) -> ti.f32:
    """
    @taichi.func

    Select the correct derivative for the upwind derivative of erosion.

    Args:
        `d_forward`: derivative in the forward direction.
        `d_backward`: derivative in the backward direction.
          
    Returns:
        derivative in the correct direction.
    """
    return ti.math.max(-d_forward, d_backward, 0) * (-1.)**(-d_forward >= d_backward)

# Switches

@ti.func
def S_ε(
    x: ti.f32,
    ε: ti.f32
) -> ti.f32:
    """
    @taichi.func
    
    Compute Sε, the regularised signum as given by K. Schaefer and J. Weickert
    in Eq (7) in [1].

    Args:
        `x`: scalar to pass through regularised signum, taking values greater 
          than 0.
        `ε`: regularisation parameter, taking values greater than 0.

    Returns:
        ti.f32 of S`ε`(`x`).

    References:
        [1]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    return (2 / ti.math.pi) * ti.math.atan2(x, ε)

@ti.func
def g_scalar(
    s_squared: ti.f32, 
    λ: ti.f32
) -> ti.f32:
    """
    @taichi.func
    
    Compute g, the function that switches between diffusion and shock, given
    by Eq. (5) in [1] by K. Schaefer and J. Weickert.

    Args:
        `s_squared`: square of some scalar, taking values greater than 0.
        `λ`: contrast parameter, taking values greater than 0.

    Returns:
        ti.f32 of g(`s_squared`).

    References:
        [1]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    return 1 / ti.math.sqrt(1 + s_squared / λ**2)

# Quality Measures

@ti.kernel
def compute_PSNR(
    denoised: ti.template(),
    ground_truth: ti.template(),
    max_val: ti.f32
) -> ti.f32:
    """
    @taichi.kernel

    Compute the Peak Signal-to-Noise Ratio (PSNR) of `u` with respect to
    `ground_truth`.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) reconstruction of image.
        `ground_truth`: ti.field(dtype=[float], shape=[Nx, Ny]), image that is
          being reconstructed.
        `MAX`: maximum possible value a pixel can take. Typically 1 or 255.

    Returns:
        PSNR in dB of `u` with respect to `ground_truth`.
    """
    shape = ground_truth.shape
    N = shape[0] * shape[1]

    squared_difference = 0.
    for I in ti.grouped(denoised):
        squared_difference += (denoised[I] - ground_truth[I])**2
    mean_squared_difference = squared_difference / N

    # PSNR is in dB, so we log_10; log_10(x) = ln(x) / ln(10).
    return 10 * ti.log(max_val**2 / mean_squared_difference) / ti.log(10)

@ti.kernel
def compute_L2(
    denoised: ti.template(),
    ground_truth: ti.template()
) -> ti.f32:
    shape = ground_truth.shape
    N = shape[0] * shape[1]
    
    squared_difference = 0.
    for I in ti.grouped(denoised):
        squared_difference += (denoised[I] - ground_truth[I])**2
    mean_squared_difference = squared_difference / N

    return ti.sqrt(mean_squared_difference)

@ti.kernel
def compute_L1(
    denoised: ti.template(),
    ground_truth: ti.template()
) -> ti.f32:
    shape = ground_truth.shape
    N = shape[0] * shape[1]
    
    absolute_difference = 0.
    for I in ti.grouped(denoised):
        absolute_difference += ti.abs(denoised[I] - ground_truth[I])
    mean_absolute_difference = absolute_difference / N

    return mean_absolute_difference

# Initialisation

@ti.kernel
def apply_boundary_conditions(
    u: ti.template(), 
    boundarypoints: ti.template(), 
    boundaryvalues: ti.template()
):
    """
    @taichi.kernel

    Apply `boundaryvalues` at `boundarypoints` to `u`.

    Args:
      Static:
        `boundarypoints`: ti.Vector.field(n=dim, dtype=[int], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.
        `boundaryvalues`: ti.Vector.field(n=dim, dtype=[float], shape=N_points),
          where `N_points` is the number of boundary points and `dim` is the 
          dimension of `u`.
      Mutated:
        `u`: ti.field(dtype=[float]) to which the boundary conditions should be 
          applied.
    """
    for I in ti.grouped(boundarypoints):
        u[boundarypoints[I]] = boundaryvalues[I]