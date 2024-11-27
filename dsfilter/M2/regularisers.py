"""
    regularisers
    ============

    Provides tools to regularise scalar fields on M_2, namely:
      1. `convolve_with_kernel_x_dir`: convolve a field with a 1D kernel along
      the x-direction.
      2. `convolve_with_kernel_y_dir`: convolve a field with a 1D kernel along
      the y-direction.
      3. `convolve_with_kernel_θ_dir`: convolve a field with a 1D kernel along
      the θ-direction.
      4. `gaussian_kernel`: computes 1D Gaussian kernels using an algorithm
      based on the DIPlib[2] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.
    We use that the spatially isotropic diffusion equation on SE(2) can be
    solved by convolving in the x-, y-, and θ-direction with some 1D kernel. For
    the x- and y-directions, this kernel is a Gaussian; for the θ-direction the
    kernel looks like a Gaussian if the amount of diffusion is sufficiently
    small.

    References:
      [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
      E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
      M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
      J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
      and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
      [2]: G. Bellaard, D.L.J. Bon, G. Pai, B.M.N. Smets, and R. Duits.
      "Analysis of (sub-)Riemannian PDE-G-CNNs". In: Journal of Mathematical
      Imaging and Vision 65 (2023), pp. 819--843.
      DOI:10.1007/s10851-023-01147-w.
"""

import taichi as ti
from dsfilter.M2.utils import (
    mirror_spatially_on_grid
)

# Scalar Field Regularisation
## Isotropic

def gaussian_kernel(σ, truncate=5., dxy=1.):
    """Compute kernel for 1D Gaussian derivative at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
        `σ`: scale of Gaussian, taking values greater than 0.
        `truncate`: number of scales `σ` at which kernel is truncated, taking 
          values greater than 0.
        `dxy`: step size in x and y direction, taking values greater than 0.

    Returns:
        Tuple ti.field(dtype=[float], shape=2*radius+1) of the Gaussian kernel
          and the radius of the kernel.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    radius = int(σ * truncate / dxy + 0.5)
    k = ti.field(dtype=ti.f32, shape=2*radius+1)
    gaussian_kernel_ti(σ, radius, k)
    return k, radius

@ti.kernel
def gaussian_kernel_ti(
    σ: ti.f32,
    radius: ti.i32,
    k: ti.template()
):
    """
    @taichi.kernel
    
    Compute 1D Gaussian kernel at scale `σ`.

    Based on the DIPlib[1] algorithm MakeHalfGaussian: https://github.com/DIPlib/diplib/blob/a6f825a69109ae388c5f0c14e76cdb2505da4594/src/linear/gauss.cpp#L95.

    Args:
      Static:
        `σ`: scale of Gaussian, taking values greater than 0.
        `radius`: radius at which kernel is truncated, taking integer values
          greater than 0.
      Mutated:
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel, which is
          updated in place.

    References:
        [1]: C. Luengo, W. Caarls, R. Ligteringen, E. Schuitema, Y. Guo,
          E. Wernersson, F. Malmberg, S. Lokhorst, M. Wolff, G. van Kempen,
          M. van Ginkel, L. van Vliet, B. Rieger, B. Verwer, H. Netten,
          J. W. Brandenburg, J. Dijk, N. van den Brink, F. Faas, K. van Wijk,
          and T. Pham. "DIPlib 3". GitHub: https://github.com/DIPlib/diplib.
    """
    for i in range(2*radius+1):
        x = -radius + i
        val = ti.math.exp(-x**2 / (2 * σ**2))
        k[i] = val
    normalise_field(k, 1)

@ti.func
def convolve_with_kernel_x_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the x-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
        of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = mirror_spatially_on_grid(ti.Vector([x - radius + i, y, θ], dt=ti.i32), u)
            s += u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_with_kernel_y_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the y-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            index = mirror_spatially_on_grid(ti.Vector([x, y - radius + i, θ], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s

@ti.func
def convolve_with_kernel_θ_dir(
    u: ti.template(),
    k: ti.template(),
    radius: ti.i32,
    u_convolved: ti.template()
):
    """
    @taichi.func
    
    Convolve `u` with the 1D kernel `k` in the θ-direction.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) array to be convolved.
        `k`: ti.field(dtype=[float], shape=2*`radius`+1) of kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
      Mutated:
        `u_convolved`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) convolution
          of `u` with `k`.
    """
    for x, y, θ in u_convolved:
        s = 0.
        for i in range(2*radius+1):
            # This may in fact give the correct convolution...
            index = mirror_spatially_on_grid(ti.Vector([x, y, θ - radius + i], dt=ti.i32), u)
            s+= u[index] * k[2*radius-i]
        u_convolved[x, y, θ] = s


# Helper Functions

@ti.func
def normalise_field(
    field: ti.template(),
    norm: ti.f32
):
    """
    @ti.func

    Normalise `field` to sum to `norm`.

    Args:
      Static:
        `norm`: desired norm for `field`, taking values greater than 0.
      Mutated:
        `field`: ti.field that is to be normalised, which is updated in place.    
    """
    current_norm = 0.
    for I in ti.grouped(field):
        ti.atomic_add(current_norm, field[I])
    norm_factor = norm / current_norm
    for I in ti.grouped(field):
        field[I] *= norm_factor