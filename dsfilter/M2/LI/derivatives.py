"""
derivatives
===========

Provides a variety of derivative operators on M_2, namely:
  1. `laplacian`: computes an approximation of the 0 Lie-Cartan Laplacian
  given some left-invariant metric tensor field, see Eq. (9) in [1].
  2. `morphological`: computes approximations of the dilation and erosion
  operators +/- ||grad u||, see Eq. (10) in [1].
  3. `TV`: computes an approximation of the total roto-translational
  generator div(grad u/||grad u||).

References:
  [1]: F.M. Sherry, K. Schaefer, and R. Duits.
  "Diffusion-Shock Filtering on the Space of Positions and Orientations."
  In: Scale Space and Variational Methods in Computer Vision (2025),
  pp. 205--217.
  DOI:10.1007/978-3-031-92369-2_16.
  [2]: A. Chambolle and Th. Pock.
  "Total roto-translational variation." In: Numerische Mathematik (2019),
  pp. 611--666.
  DOI:10.1007/s00211-019-01026-w.
  [3]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
  "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
  Positions and Orientations." In: Journal of Mathematical Imaging and
  Vision (2021), pp. 237-262.
  DOI:10.1007/s10851-020-00991-4.
"""

import taichi as ti
from dsfilter.M2.utils import scalar_trilinear_interpolate
from dsfilter.M2.regularisers import (
    convolve_with_kernel_x_dir,
    convolve_with_kernel_y_dir,
    convolve_with_kernel_θ_dir,
)
from dsfilter.utils import (
    select_upwind_derivative_dilation,
    select_upwind_derivative_erosion,
)

# Actual Derivatives


@ti.kernel
def laplacian(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    laplacian_u: ti.template(),
):
    """
    @taichi.kernel

    Compute an approximation of the Laplace-Beltrami operator applied to `u`
    using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) constants of the inverse of
          the diagonal metric tensor with respect to left invariant basis.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) laplacian of
          u, which is updated in place.
    """
    I_A3 = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(laplacian_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A11 = (
            scalar_trilinear_interpolate(u, I + I_A1)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A1)
        ) / dxy**2
        A22 = (
            scalar_trilinear_interpolate(u, I + I_A2)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A2)
        ) / dxy**2
        A33 = (
            scalar_trilinear_interpolate(u, I + I_A3)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A3)
        ) / dθ**2
        # Δu = div(grad(u)) = sqrt(det(g)) A_i (sqrt(det(g)) g^ij A_j u) = g^ij A_i A_j u = g^ii A_i A_i u
        laplacian_u[I] = G_inv[0] * A11 + G_inv[1] * A22 + G_inv[2] * A33


@ti.kernel
def laplacian_s(
    u: ti.template(),
    G_inv: ti.types.vector(2, ti.f32),
    dxy: ti.f32,
    θs: ti.template(),
    laplacian_u: ti.template(),
):
    """
    @taichi.kernel

    Compute an approximation of the spatial Laplace-Beltrami operator applied to
    `u` using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `G_inv`: ti.types.vector(n=2, dtype=[float]) spatial constants of the
          inverse of the diagonal metric tensor with respect to left invariant basis.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) spatial
          laplacian of u, which is updated in place.
    """
    for I in ti.grouped(laplacian_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A11 = (
            scalar_trilinear_interpolate(u, I + I_A1)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A1)
        ) / dxy**2
        A22 = (
            scalar_trilinear_interpolate(u, I + I_A2)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A2)
        ) / dxy**2
        # Δu = div(grad(u)) = sqrt(det(g)) A_i (sqrt(det(g)) g^ij A_j u) = g^ij A_i A_j u = g^ii A_i A_i u
        laplacian_u[I] = G_inv[0] * A11 + G_inv[1] * A22


@ti.kernel
def morphological(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    dilation_u: ti.template(),
    erosion_u: ti.template(),
):
    """
    @taichi.kernel

    Compute upwind approximations of the morphological derivatives
    +/- ||grad `u`||.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2, Nθ]) which we want to
          differentiate.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) constants of the inverse of
          the diagonal metric tensor with respect to left invariant basis.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) -||grad `u`||,
          which is updated in place.
    """
    I_A3 = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(dilation_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A1_forward = (scalar_trilinear_interpolate(u, I + I_A1) - u[I]) / dxy
        A2_forward = (scalar_trilinear_interpolate(u, I + I_A2) - u[I]) / dxy
        A3_forward = (scalar_trilinear_interpolate(u, I + I_A3) - u[I]) / dθ
        A1_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A1)) / dxy
        A2_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A2)) / dxy
        A3_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A3)) / dθ

        # ||grad u|| = sqrt(G(grad u, grad u)) = sqrt(g^ij A_i u A_j u) = sqrt(g^ii (A_i u)^2)
        # Dilation
        dilation_u[I] = ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_dilation(A1_forward, A1_backward) ** 2
            + G_inv[1] * select_upwind_derivative_dilation(A2_forward, A2_backward) ** 2
            + G_inv[2] * select_upwind_derivative_dilation(A3_forward, A3_backward) ** 2
        )
        # Erosion
        erosion_u[I] = -ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_erosion(A1_forward, A1_backward) ** 2
            + G_inv[1] * select_upwind_derivative_erosion(A2_forward, A2_backward) ** 2
            + G_inv[2] * select_upwind_derivative_erosion(A3_forward, A3_backward) ** 2
        )


@ti.kernel
def morphological_s(
    u: ti.template(),
    G_inv: ti.types.vector(2, ti.f32),
    dxy: ti.f32,
    θs: ti.template(),
    dilation_u: ti.template(),
    erosion_u: ti.template(),
):
    """
    @taichi.kernel

    Compute upwind approximations of the spatial morphological derivatives
    +/- ||grad `u`||.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2, Nθ]) which we want to
          differentiate.
        `G_inv`: ti.types.vector(n=2, dtype=[float]) spatial constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis.
        `θs`: angle coordinate at each grid point.
        `dxy`: step size in x and y direction, taking values greater than 0.
      Mutated:
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) -||grad `u`||,
          which is updated in place.
    """
    for I in ti.grouped(dilation_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)

        A1_forward = (scalar_trilinear_interpolate(u, I + I_A1) - u[I]) / dxy
        A2_forward = (scalar_trilinear_interpolate(u, I + I_A2) - u[I]) / dxy
        A1_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A1)) / dxy
        A2_backward = (u[I] - scalar_trilinear_interpolate(u, I - I_A2)) / dxy

        # ||grad u|| = sqrt(G(grad u, grad u)) = sqrt(g^ij A_i u A_j u) = sqrt(g^ii (A_i u)^2)
        # Dilation
        dilation_u[I] = ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_dilation(A1_forward, A1_backward) ** 2
            + G_inv[1] * select_upwind_derivative_dilation(A2_forward, A2_backward) ** 2
        )
        # Erosion
        erosion_u[I] = -ti.math.sqrt(
            G_inv[0] * select_upwind_derivative_erosion(A1_forward, A1_backward) ** 2
            + G_inv[1] * select_upwind_derivative_erosion(A2_forward, A2_backward) ** 2
        )


@ti.func
def gradient(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ξ: ti.f32,
    gradient_u: ti.template(),
):
    """
    @taichi.func

    Compute an approximation of the gradient of `u` using central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
      Mutated:
        `gradient_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) gradient of u,
          which is updated in place.
    """
    I_A3 = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32) / 2
    for I in ti.grouped(gradient_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32) / 2
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32) / 2
        # ||grad u|| = sqrt(G(grad u, grad u)) = sqrt(g^ij A_i u A_j u) = sqrt(ξ^-2 ((A_1 u)^2 + (A_2 u)^2) + (A_3 u)^2)
        gradient_u[I] = ti.math.sqrt(
            (
                (
                    scalar_trilinear_interpolate(u, I + I_A1)
                    - scalar_trilinear_interpolate(u, I - I_A1)
                )
                / (ξ * dxy)
            )
            ** 2
            + (
                (
                    scalar_trilinear_interpolate(u, I + I_A2)
                    - scalar_trilinear_interpolate(u, I - I_A2)
                )
                / (ξ * dxy)
            )
            ** 2
            + (
                (
                    scalar_trilinear_interpolate(u, I + I_A3)
                    - scalar_trilinear_interpolate(u, I - I_A3)
                )
                / dθ
            )
            ** 2
        )


@ti.func
def gradient_s(
    u: ti.template(), dxy: ti.f32, θs: ti.template(), gradient_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the spatial gradient of `u` using
    central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `gradient_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) spatial
          gradient of u, which is updated in place.
    """
    for I in ti.grouped(gradient_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32) / 2
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32) / 2
        # ||grad u|| = sqrt(G(grad u, grad u)) = sqrt(g^ij A_i u A_j u) = sqrt(g^11 (A_1 u)^2 + g^22 (A_2 u)^2)
        gradient_u[I] = ti.math.sqrt(
            (
                (
                    scalar_trilinear_interpolate(u, I + I_A1)
                    - scalar_trilinear_interpolate(u, I - I_A1)
                )
                / dxy
            )
            ** 2
            + (
                (
                    scalar_trilinear_interpolate(u, I + I_A2)
                    - scalar_trilinear_interpolate(u, I - I_A2)
                )
                / dxy
            )
            ** 2
        )


@ti.func
def laplace_perp(
    u: ti.template(),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    ξ: ti.f32,
    laplace_perp_u: ti.template(),
):
    """
    @taichi.func

    Compute an approximation of the perpendicular laplacian of `u` using central
    differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
      Mutated:
        `laplace_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          perpendicular laplacian of u, which is updated in place.
    """
    I_A3 = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32)
    for I in ti.grouped(laplace_perp_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)
        # Δ_perp u = div_perp(grad_perp(u)) = sqrt(det(g)) A_i (sqrt(det(g)) g^ij A_j u) = g^ij A_i A_j u = A_2 A_2 u + ξ^2 A_3 A_3 u
        laplace_perp_u[I] = (
            scalar_trilinear_interpolate(u, I + I_A2)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A2)
        ) / (ξ * dxy) ** 2 + (
            scalar_trilinear_interpolate(u, I + I_A3)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A3)
        ) / dθ**2


@ti.func
def laplace_perp_s(
    u: ti.template(), dxy: ti.f32, θs: ti.template(), laplace_perp_u: ti.template()
):
    """
    @taichi.func

    Compute an approximation of the spatial perpendicular laplacian of `u` using
    central differences.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `laplace_perp_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ])
          perpendicular laplacian of u, which is updated in place.
    """
    for I in ti.grouped(laplace_perp_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32)
        # Δ_perp u = div_perp(grad_perp(u)) = sqrt(det(g)) A_i (sqrt(det(g)) g^ij A_j u) = g^ij A_i A_j u = A_2 A_2 u
        laplace_perp_u[I] = (
            scalar_trilinear_interpolate(u, I + I_A2)
            - 2 * u[I]
            + scalar_trilinear_interpolate(u, I - I_A2)
        ) / dxy**2


@ti.kernel
def TV(
    u: ti.template(),
    G_inv: ti.types.vector(3, ti.f32),
    dxy: ti.f32,
    dθ: ti.f32,
    θs: ti.template(),
    k_s: ti.template(),
    radius_s: ti.template(),
    k_o: ti.template(),
    radius_o: ti.template(),
    A1_u: ti.template(),
    A2_u: ti.template(),
    A3_u: ti.template(),
    grad_norm_u: ti.template(),
    normalised_grad_1: ti.template(),
    normalised_grad_2: ti.template(),
    normalised_grad_3: ti.template(),
    TV_u: ti.template(),
    storage: ti.template(),
):
    """
    @taichi.kernel

    Compute an approximation of the Total Roto-Translational Variation (TR-TV)
    operator applied to `u` using central differences.[1]

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to
          differentiate.
        `G_inv`: ti.types.vector(n=3, dtype=[float]) constants of the inverse of
          the diagonal metric tensor with respect to left invariant basis.
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in orientational direction, taking values greater than
          0.
        `θs`: angle coordinate at each grid point.
      Mutated:
        `TV_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) laplacian of
          u, which is updated in place.

    References:
        [1]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
          "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
          Positions and Orientations." In: Journal of Mathematical Imaging and
          Vision (2021), pp. 237-262.
          DOI:10.1007/s10851-020-00991-4.
    """
    I_A3 = ti.Vector([0.0, 0.0, 1.0], dt=ti.f32) / 2
    for I in ti.grouped(A1_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32) / 2
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32) / 2

        c1 = (
            scalar_trilinear_interpolate(u, I + I_A1)
            - scalar_trilinear_interpolate(u, I - I_A1)
        ) / dxy
        c2 = (
            scalar_trilinear_interpolate(u, I + I_A2)
            - scalar_trilinear_interpolate(u, I - I_A2)
        ) / dxy
        c3 = (
            scalar_trilinear_interpolate(u, I + I_A3)
            - scalar_trilinear_interpolate(u, I - I_A3)
        ) / dθ
        A1_u[I] = c1
        A2_u[I] = c2
        A3_u[I] = c3
        grad_norm_u[I] = ti.math.sqrt(
            G_inv[0] * c1**2 + G_inv[1] * c2**2 + G_inv[2] * c3**2
        )
    convolve_with_kernel_x_dir(grad_norm_u, k_s, radius_s, TV_u)
    convolve_with_kernel_y_dir(TV_u, k_s, radius_s, storage)
    convolve_with_kernel_θ_dir(storage, k_o, radius_o, grad_norm_u)

    for I in ti.grouped(normalised_grad_1):
        normalised_grad_1[I] = G_inv[0] * A1_u[I] / grad_norm_u[I]
        normalised_grad_2[I] = G_inv[1] * A2_u[I] / grad_norm_u[I]
        normalised_grad_3[I] = G_inv[2] * A3_u[I] / grad_norm_u[I]

    for I in ti.grouped(TV_u):
        θ = θs[I]
        cos = ti.math.cos(θ)
        sin = ti.math.sin(θ)
        I_A1 = ti.Vector([cos, sin, 0.0], dt=ti.f32) / 2
        I_A2 = ti.Vector([-sin, cos, 0.0], dt=ti.f32) / 2

        divnormgrad1 = (
            scalar_trilinear_interpolate(normalised_grad_1, I + I_A1)
            - scalar_trilinear_interpolate(normalised_grad_1, I - I_A1)
        ) / dxy
        divnormgrad2 = (
            scalar_trilinear_interpolate(normalised_grad_2, I + I_A2)
            - scalar_trilinear_interpolate(normalised_grad_2, I - I_A2)
        ) / dxy
        divnormgrad3 = (
            scalar_trilinear_interpolate(normalised_grad_3, I + I_A3)
            - scalar_trilinear_interpolate(normalised_grad_3, I - I_A3)
        ) / dθ
        TV_u[I] = divnormgrad1 + divnormgrad2 + divnormgrad3
