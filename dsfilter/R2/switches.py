"""
    switches
    ========

    Provides the operators to switch between diffusion and shock, and between
    dilation and erosion, as described by K. Schaefer and J. Weickert.[1][2]
    The primary methods are:
      1. `DS_switch`: switches between diffusion and shock. If there is locally
      a clear orientation, more shock is applied, see Eq. (7) in [1].
      2. `morphological_switch`: switches between dilation and erosion. If the
      data is locally convex, erosion is applied, while if the data is locally
      concave, dilation is applied, see Eq. (4) in [1].

    References:
      [1]: K. Schaefer and J. Weickert.
      "Diffusion-Shock Inpainting." In: Scale Space and Variational Methods in
      Computer Vision 14009 (2023), pp. 588--600.
      DOI:10.1007/978-3-031-31975-4_45.
      [2]: K. Schaefer and J. Weickert.
      "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
      Imaging and Vision (2024).
      DOI:10.1007/s10851-024-01175-0.
"""

import taichi as ti
from dsfilter.R2.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir
)
from dsfilter.R2.derivatives import central_derivatives_second_order
from dsfilter.R2.utils import sanitize_index
from dsfilter.utils import (
    S_ε,
    g_scalar
)

# Diffusion-Shock

## Switcher

@ti.kernel
def DS_switch(
    u: ti.template(),
    dxy: ti.f32,
    k: ti.template(),
    radius: ti.i32,
    λ: ti.f32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    switch: ti.template(),
    convolution_storage: ti.template()
):
    """
    @taichi.kernel

    Determine to what degree we should perform diffusion or shock, as described
    by K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `k`: ti.field(dtype=ti.f32, shape=2*`radius`+1) Gaussian kernel.
        `radius`: radius at which kernel `k` is truncated, taking integer values
          greater than 0.
        `λ`: contrast parameter, taking values greater than 0.
      Mutated:
        `d_d*`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of derivatives, which are
          updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) values that
          determine the degree of diffusion or shock, taking values between 0
          and 1, which is updated in place.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting." In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1007/978-3-031-31975-4_45.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k, radius, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k, radius, switch)
    # Then compute gradient with Sobel operators.
    sobel_gradient(switch, dxy, d_dx, d_dy)
    for I in ti.grouped(switch):
        switch[I] = g_scalar(d_dx[I]**2 + d_dy[I]**2, λ)

# Morphological

## Switcher

@ti.kernel
def morphological_switch(
    u: ti.template(),
    u_σ: ti.template(),
    dxy: ti.f32,
    ε: ti.f32,
    k_int: ti.template(),
    radius_int: ti.i32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    k_ext: ti.template(),
    radius_ext: ti.template(),
    Jρ_storage: ti.template(),
    Jρ11: ti.template(),
    Jρ12: ti.template(),
    Jρ22: ti.template(),
    d_dxx: ti.template(),
    d_dxy: ti.template(),
    d_dyy: ti.template(),
    switch: ti.template(),
    convolution_storage: ti.template()
):
    """
    @taichi.func
    
    Determine whether to perform dilation or erosion, as described by
    K. Schaefer and J. Weickert.[1][2]

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion, taking values greater than 0.
        `k_int`: ti.field(dtype=[float], shape=2*`radius_int`+1) Gaussian kernel
          with standard deviation σ.
        `radius_int`: radius at which kernel `k_int` is truncated, taking
          integer values greater than 0.
        `k_ext`: ti.field(dtype=[float], shape=2*`radius_ext`+1) Gaussian kernel
          with standard deviation ρ.
        `radius_ext`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `u_σ`: ti.field(dtype=[float], shape=[Nx, Ny]) u convolved with Gaussian
          with standard deviation σ.
        `d_d*`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of first order Gaussian
          derivatives, which are updated in place.
        `Jρ_storage`: ti.field(dtype=[float], shape=[Nx, Ny]) array to hold
          intermediate computations for the structure tensor.
        `Jρ**`: ti.field(dtype=[float], shape=[Nx, Ny]) **-component of the
          regularised structure tensor.
        `d_d**`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of second order Gaussian
          derivatives, which are updated in place.
        `switch`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) values that
          determine the degree of dilation or erosion, taking values between -1
          and 1, which is updated in place.
        `convolution_storage`: ti.field(dtype=[float], shape=[Nx, Ny]) array to
          hold intermediate results when performing convolutions.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Diffusion-Shock Inpainting." In: Scale Space and Variational Methods
          in Computer Vision 14009 (2023), pp. 588--600.
          DOI:10.1007/978-3-031-31975-4_45.
        [2]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    compute_structure_tensor(u, u_σ, dxy, k_int, radius_int, d_dx, d_dy, k_ext, radius_ext, Jρ_storage, Jρ11, Jρ12,
                             Jρ22, convolution_storage)
    # Regularise with same Gaussian kernel as when computing gradient for
    # structure tensor.
    convolve_with_kernel_x_dir(u, k_int, radius_int, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_int, radius_int, switch)
    # Compute second derivatives of u_σ.
    central_derivatives_second_order(switch, dxy, d_dxx, d_dxy, d_dyy)
    # Compute second derivative of u_σ in the direction of the dominant
    # eigenvector.
    for I in ti.grouped(switch):
        A11 = Jρ11[I]
        A12 = Jρ12[I]
        A22 = Jρ22[I]
        # The dominant eigenvector of a symmetrix 2x2 matrix A with nonnegative
        # trace A11 + A22, such as the structure tensor, is given by
        #   (-(-A11 + A22 - sqrt((A11 - A22)**2 + 4 A12**2)), 2 A12).
        v1 = -(-A11 + A22 - ti.math.sqrt((A11 - A22)**2 + 4 * A12**2))
        norm = ti.math.sqrt(v1**2 + (2 * A12)**2) + 10**-8

        c = v1 / norm
        s = 2 * A12 / norm

        d_dww = c**2 * d_dxx[I] + 2 * c * s * d_dxy[I] + s**2 * d_dyy[I]
        switch[I] = (ε > 0.) * S_ε(d_dww, ε) + (ε == 0.) * ti.math.sign(d_dww)

@ti.func
def compute_structure_tensor(
    u: ti.template(),
    u_σ: ti.template(),
    dxy: ti.f32,
    k_int: ti.template(),
    radius_int: ti.i32,
    d_dx: ti.template(),
    d_dy: ti.template(),
    k_ext: ti.template(),
    radius_ext: ti.i32,
    Jρ_storage: ti.template(),
    Jρ11: ti.template(),
    Jρ12: ti.template(),
    Jρ22: ti.template(),
    convolution_storage: ti.template()
):
    """
    @taichi.func

    Compute the structure tensor. 

    Args:
      Static:
        `u`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) current state.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `k_int`: ti.field(dtype=[float], shape=2*`radius_int`+1) Gaussian kernel
          with standard deviation σ.
        `radius_int`: radius at which kernel `k_int` is truncated, taking
          integer values greater than 0.
        `k_ext`: ti.field(dtype=[float], shape=2*`radius_ext`+1) first order
          Gaussian derivative kernel.
        `radius_ext`: radius at which kernel `k_ext` is truncated, taking
          integer values greater than 0.
      Mutated:
        `u_σ`: ti.field(dtype=[float], shape=[Nx, Ny]) u convolved with Gaussian
          with standard deviation σ.
        `d_d*`: ti.field(dtype=[float], shape=[Nx, Ny]) Gaussian derivatives,
          which are updated in place.
        `Jρ_storage`: ti.field(dtype=[float], shape=[Nx, Ny]) array to hold
          intermediate computations for the structure tensor.
        `Jρ**`: ti.field(dtype=[float], shape=[Nx, Ny]) **-component of the
          regularised structure tensor.
        `convolution_storage`: ti.field(dtype=[float], shape=[Nx, Ny]) array to
          hold intermediate results when performing convolutions.
    """
    # First regularise with Gaussian convolution.
    convolve_with_kernel_x_dir(u, k_int, radius_int, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_int, radius_int, u_σ)
    # Then compute gradient with Sobel operators.
    sobel_gradient(u_σ, dxy, d_dx, d_dy)
    # Compute Jρ_11.
    for I in ti.grouped(Jρ_storage):
        Jρ_storage[I] = d_dx[I]**2
    convolve_with_kernel_x_dir(Jρ_storage, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, Jρ11)
    # Compute Jρ_12.
    for I in ti.grouped(Jρ_storage):
        Jρ_storage[I] = d_dx[I] * d_dy[I]
    convolve_with_kernel_x_dir(Jρ_storage, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, Jρ12)
    # Compute Jρ_22.
    for I in ti.grouped(Jρ_storage):
        Jρ_storage[I] = d_dy[I]**2
    convolve_with_kernel_x_dir(Jρ_storage, k_ext, radius_ext, convolution_storage)
    convolve_with_kernel_y_dir(convolution_storage, k_ext, radius_ext, Jρ22)

# Derivatives

@ti.func
def sobel_gradient(
    u: ti.template(),
    dxy: ti.f32,
    dx_u: ti.template(),
    dy_u: ti.template()
):
    """
    @taichi.func
    
    Compute approximations of the first order derivatives of `u` in the x and y 
    direction using Sobel operators, as described in Eq. (26) in [1] by
    K. Schaefer and J. Weickert.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny]) which we want to 
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `d*_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of d* `u`, which is
          updated in place.

    References:
        [1]: K. Schaefer and J. Weickert.
          "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
          Imaging and Vision (2024).
          DOI:10.1007/s10851-024-01175-0.
    """
    I_dx = ti.Vector([1, 0], dt=ti.i32)
    I_dy = ti.Vector([0, 1], dt=ti.i32)
    I_dplus = I_dx + I_dy  # Positive diagonal
    I_dminus = I_dx - I_dy # Negative diagonal
    for I in ti.grouped(u):
        I_dx_forward = sanitize_index(I + I_dx, u)
        I_dx_backward = sanitize_index(I - I_dx, u)
        I_dy_forward = sanitize_index(I + I_dy, u)
        I_dy_backward = sanitize_index(I - I_dy, u)
        I_dplus_forward = sanitize_index(I + I_dplus, u)
        I_dplus_backward = sanitize_index(I - I_dplus, u)
        I_dminus_forward = sanitize_index(I + I_dminus, u)
        I_dminus_backward = sanitize_index(I - I_dminus, u)
        # du/dx Stencil
        # -1 | 0 | 1
        # -2 | 0 | 2
        # -1 | 0 | 1
        dx_u[I] = (
            -1 * u[I_dminus_backward] +
            -2 * u[I_dx_backward] +
            -1 * u[I_dplus_backward] +
            1 * u[I_dminus_forward] +
            2 * u[I_dx_forward] +
            1 * u[I_dplus_forward]
        ) / (8 * dxy)
        # du/dy Stencil
        #  1 |  2 |  1
        #  0 |  0 |  0
        # -1 | -2 | -1
        dy_u[I] = (
            -1 * u[I_dplus_backward] +
            -2 * u[I_dy_backward] +
            -1 * u[I_dminus_forward] +
            1 * u[I_dminus_backward] +
            2 * u[I_dy_forward] +
            1 * u[I_dplus_forward]
        ) / (8 * dxy)