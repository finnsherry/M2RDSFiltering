"""
    filter
    ======

    Provides methods to apply left-invariant M_2 Diffusion-Shock filtering,[1]
    inspired by the Diffusion-Shock filtering on R^2 by K. Schaefer and
    J. Weickert.[2][3]
    The primary methods are:
      1. `DS_enhancing_LI`: perform left-invariant RDS filtering on M_2 for
      denoising.
      2. `DS_inpainting_LI`: perform left-invariant RDS inpainting on M_2.
      3. `TV_enhancing_LI`: perform left-invariant TR-TV flow on M_2 for
      denoising.[4][5]

    References:
      [1]: F.M. Sherry, K. Schaefer, and R. Duits.
      "Diffusion-Shock Filtering on the Space of Positions and Orientations."
      In: Scale Space and Variational Methods in Computer Vision (2025), pp. .
      DOI:.
      [2]: K. Schaefer and J. Weickert.
      "Diffusion-Shock Inpainting." In: Scale Space and Variational Methods in
      Computer Vision 14009 (2023), pp. 588--600.
      DOI:10.1007/978-3-031-31975-4_45.
      [3]: K. Schaefer and J. Weickert.
      "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
      Imaging and Vision (2024).
      DOI:10.1007/s10851-024-01175-0.
      [4]: A. Chambolle and Th. Pock.
      "Total roto-translational variation." In: Numerische Mathematik (2019),
      pp. 611--666.
      DOI:10.1137/s00211-019-01026-w.
      [5]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
      "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
      Positions and Orientations." In: Journal of Mathematical Imaging and
      Vision (2021).
      DOI:10.1007/s10851-020-00991-4.
"""

import taichi as ti
import numpy as np
from tqdm import tqdm
from dsfilter.M2.LI.switches import (
    DS_switch,
    DS_switch_s,
    morphological_switch,
    morphological_switch_s
)
from dsfilter.M2.LI.derivatives import (
    laplacian,
    laplacian_s,
    morphological,
    morphological_s,
    TV
)
from dsfilter.M2.regularisers import gaussian_kernel
from dsfilter.M2.utils import project_down
from dsfilter.utils import (
    compute_PSNR,
    compute_L2,
    compute_L1
)

def DS_enhancing(u0_np, ground_truth_np, θs_np, ξ, T, G_D_inv_np, G_S_inv_np, σ, ρ, ν, λ, ε=0., dxy=1.):
    """
    Perform left-invariant Diffusion-Shock filtering in M_2 for denoising.[1]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `ground_truth_np`: np.ndarray ground truth, with shape [Nx, Ny].
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
        `T`: time that image is evolved under the DS PDE.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
        `σ`: standard deviation in the spatial-direction of the internal
          regularisation on the switch between dilation and erosion, taking
          values greater than 0.
        `ρ`: standard deviation in the spatial-direction of the external
          regularisation on the switch between dilation and erosion, taking
          values greater than 0.
        `ν`: standard deviation in the spatial-direction of the internal
          regularisation on the switch between diffusion and shock, taking
          values greater than 0.
        `λ`: contrast parameter used to determine whether to perform diffusion
          or shock based on the degree of local orientation.
        
      Optional:
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion.
        `dxy`: size of pixels in the x- and y-directions. Defaults to 1.

    Returns:
        np.ndarray solution to the DS PDE with initial condition `u0_np` at
        time `T`.

    References:
        [1]: F.M. Sherry, K. Schaefer, and R. Duits.
          "Diffusion-Shock Filtering on the Space of Positions and
          Orientations." In: Scale Space and Variational Methods in Computer
          Vision (2025), pp. .
          DOI:.
    """
    # Set hyperparameters
    shape = u0_np.shape
    _, _, Nθ = shape
    dθ = 2 * np.pi / Nθ
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
    n = int(T / dt)

    k_s_DS, radius_s_DS = gaussian_kernel(ν, dxy=dxy)
    k_o_DS, radius_o_DS = gaussian_kernel(ν * ξ, dxy=dθ)
    k_s_morph_int, radius_s_morph_int = gaussian_kernel(σ, dxy=dxy)
    k_o_morph_int, radius_o_morph_int = gaussian_kernel(σ * ξ, dxy=dθ)
    k_s_morph_ext, radius_s_morph_ext = gaussian_kernel(ρ, dxy=dxy)
    k_o_morph_ext, radius_o_morph_ext = gaussian_kernel(ρ * ξ, dxy=dθ)

    # Initialise TaiChi objects
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    G_D_inv = ti.Vector(G_D_inv_np, dt=ti.f32)
    G_S_inv = ti.Vector(G_S_inv_np, dt=ti.f32)
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(np.zeros_like(u0_np))
    du_dt = ti.field(dtype=ti.f32, shape=shape)

    ## Padded versions for derivatives
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    ### Laplacian
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological
    dilation_u = ti.field(dtype=ti.f32, shape=shape)
    erosion_u = ti.field(dtype=ti.f32, shape=shape)

    ## Fields for switches
    u_switch = ti.field(dtype=ti.f32, shape=shape)
    fill_u_switch(u, u_switch)
    storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    gradient_u = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    ## Image Quality Measures
    max_val = 255. # Images are assumed to take gray values in [0, 255].
    ground_truth = ti.field(dtype=ti.f32, shape=shape[:-1])
    ground_truth.from_numpy(ground_truth_np)
    u_projected = ti.field(dtype=ti.f32, shape=shape[:-1])
    project_down(u, u_projected, 0., max_val, 1.)
    PSNR = [compute_PSNR(u_projected, ground_truth, max_val)]
    L1 = [compute_L1(u_projected, ground_truth)]
    L2 = [compute_L2(u_projected, ground_truth)]

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch(u_switch, dxy, dθ, ξ, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, gradient_u, switch_DS, storage)
        morphological_switch(u_switch, dxy, dθ, ξ, θs, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
                             radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
                             laplace_perp_u, switch_morph, storage)
        # Compute derivatives
        laplacian(u, G_D_inv, dxy, dθ, θs, laplacian_u)
        morphological(u, G_S_inv, dxy, dθ, θs, dilation_u, erosion_u)
        # Step
        step_DS(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)

        project_down(u, u_projected, 0., max_val, 1.)
        PSNR.append(compute_PSNR(u_projected, ground_truth, max_val))
        L2.append(compute_L2(u_projected, ground_truth))
        L1.append(compute_L1(u_projected, ground_truth))
    return u.to_numpy(), np.array(PSNR), np.array(L2), np.array(L1)

def DS_inpainting(u0_np, mask_np, θs_np, T, G_D_inv_np, G_S_inv_np, σ_s, σ_o, ρ_s, ρ_o, ν_s, ν_o, λ, ε=0., dxy=1.):
    """
    Perform left-invariant Diffusion-Shock inpainting in M_2.[1]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `T`: time that image is evolved under the DS PDE.
        `G_D_inv_np`: np.ndarray(shape=(2,), dtype=[float]) spatial constants of
          the inverse of the diagonal metric tensor with respect to left
          invariant basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(2,), dtype=[float]) spatial constants of
          the inverse of the diagonal metric tensor with respect to left
          invariantbasis used to define the shock.
        `σ_s`: standard deviation in the spatial directions of the internal
          regularisation of the morphological switch, taking values greater than
          0.
        `σ_o`: standard deviation in the orientational directions of the
          internal regularisation of the morphological switch, taking values
          greater than 0.
        `ρ_s`: standard deviation in the spatial directions of the external
          regularisation of the morphological switch, taking values greater than
          0.
        `ρ_o`: standard deviation in the orientational directions of the
          external regularisation of the morphological switch, taking values
          greater than 0.
        `ν_s`: standard deviation in the spatial directions of the internal
          regularisation of the morphological switch, taking values greater than
          0.
        `ν_o`: standard deviation in the orientational directions of the
          internal regularisation of the diffusion-shock switch, taking values
          greater than 0.
        `λ`: contrast parameter used to determine whether to perform diffusion
          or shock based on the degree of local orientation.
        
      Optional:
        `ε`: regularisation parameter for the signum function used to switch
          between dilation and erosion.
        `dxy`: size of pixels in the x- and y-directions. Defaults to 1.

    Returns:
        np.ndarray solution to the DS PDE with initial condition `u0_np` at
        time `T`.

    References:
        [1]: F.M. Sherry, K. Schaefer, and R. Duits.
          "Diffusion-Shock Filtering on the Space of Positions and
          Orientations." In: Scale Space and Variational Methods in Computer
          Vision (2025), pp. .
          DOI:.
    """
    # Set hyperparameters
    shape = u0_np.shape
    _, _, Nθ = shape
    dθ = 2 * np.pi / Nθ
    dt = compute_timestep(dxy, dθ, G_D_inv_np, G_S_inv_np)
    n = int(T / dt)

    k_s_DS, radius_s_DS = gaussian_kernel(ν_s, dxy=dxy)
    k_o_DS, radius_o_DS = gaussian_kernel(ν_o, dxy=dθ)
    k_s_morph_int, radius_s_morph_int = gaussian_kernel(σ_s, dxy=dxy)
    k_o_morph_int, radius_o_morph_int = gaussian_kernel(σ_o, dxy=dθ)
    k_s_morph_ext, radius_s_morph_ext = gaussian_kernel(ρ_s, dxy=dxy)
    k_o_morph_ext, radius_o_morph_ext = gaussian_kernel(ρ_o, dxy=dθ)

    # Initialise TaiChi objects
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    G_D_inv = ti.Vector(G_D_inv_np, dt=ti.f32)
    G_S_inv = ti.Vector(G_S_inv_np, dt=ti.f32)
    mask = ti.field(dtype=ti.f32, shape=shape)
    mask.from_numpy(mask_np)
    du_dt = ti.field(dtype=ti.f32, shape=shape)

    ## Padded versions for derivatives
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    ### Laplacian
    laplacian_u = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological
    dilation_u = ti.field(dtype=ti.f32, shape=shape)
    erosion_u = ti.field(dtype=ti.f32, shape=shape)

    ## Fields for switches
    u_switch = ti.field(dtype=ti.f32, shape=shape)
    fill_u_switch(u, u_switch)
    storage = ti.field(dtype=ti.f32, shape=shape)
    ### DS switch
    gradient_u = ti.field(dtype=ti.f32, shape=shape)
    switch_DS = ti.field(dtype=ti.f32, shape=shape)
    ### Morphological switch
    laplace_perp_u = ti.field(dtype=ti.f32, shape=shape)
    switch_morph = ti.field(dtype=ti.f32, shape=shape)

    for _ in tqdm(range(n)):
        # Compute switches
        DS_switch_s(u_switch, dxy, θs, k_s_DS, radius_s_DS, k_o_DS, radius_o_DS, λ, gradient_u, switch_DS, storage)
        morphological_switch_s(u_switch, dxy, θs, ε, k_s_morph_int, radius_s_morph_int, k_o_morph_int,
                               radius_o_morph_int, k_s_morph_ext, radius_s_morph_ext, k_o_morph_ext, radius_o_morph_ext,
                               laplace_perp_u, switch_morph, storage)
        # Compute derivatives
        laplacian_s(u, G_D_inv, dxy, θs, laplacian_u)
        morphological_s(u, G_S_inv, dxy, θs, dilation_u, erosion_u)
        # Step
        step_DS(u, mask, dt, switch_DS, switch_morph, laplacian_u, dilation_u, erosion_u, du_dt)
        # Update fields for switches
        fill_u_switch(u, u_switch)
    return u.to_numpy()

@ti.kernel
def step_DS(
    u: ti.template(),
    mask: ti.template(),
    dt: ti.f32,
    switch_DS: ti.template(),
    switch_morph: ti.template(),
    laplacian_u: ti.template(),
    dilation_u: ti.template(),
    erosion_u: ti.template(),
    du_dt: ti.template()
):
    """
    @taichi.kernel

    Perform a single timestep of M_2 Diffusion-Shock filtering.[1]

    Args:
      Static:
        `mask`: ti.field(dtype=[float], shape=[Nx, Ny]) inpainting mask.
        `dt`: step size, taking values greater than 0.
        `switch_DS`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of diffusion or shock, taking values between 0
          and 1.
        `switch_morph`: ti.field(dtype=ti.f32, shape=[Nx, Ny]) of values that
          determine the degree of dilation or erosion, taking values between -1
          and 1.
        `laplacian_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of laplacian of
          `u`, which is updated in place.
        `dilation_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of ||grad `u`||,
          which is updated in place.
        `erosion_u`: ti.field(dtype=[float], shape=[Nx, Ny]) of -||grad `u`||,
          which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx+2, Ny+2]) which we want to evolve
          with the DS PDE.
        `du_dt`: ti.field(dtype=[float], shape=[Nx, Ny]) change in `u` in a
          single time step, not taking into account the mask.

    References:
        [1]: F.M. Sherry, K. Schaefer, and R. Duits.
          "Diffusion-Shock Filtering on the Space of Positions and
          Orientations." In: Scale Space and Variational Methods in Computer
          Vision (2025), pp. .
          DOI:.
    """
    for I in ti.grouped(du_dt):
        du_dt[I] = (
            laplacian_u[I] * switch_DS[I] +
            (1 - switch_DS[I]) * (
                # Do erosion when switch_morph = 1.
                erosion_u[I] * (switch_morph[I] > 0.) * ti.abs(switch_morph[I])  +
                # Do dilation when switch_morph = -1.
                dilation_u[I] * (switch_morph[I] < 0.) * ti.abs(switch_morph[I])
            )
        )
        u[I] += dt * du_dt[I] * (1 - mask[I]) # Only change values in the mask.

def compute_timestep(dxy, dθ, G_D_inv, G_S_inv):
    """
    Compute timestep to solve Diffusion-Shock PDE,[1] such that the scheme
    retains the maximum-minimum principle of the continuous PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
    
    Returns:
        timestep, taking values greater than 0.

    References:
        [1]: F.M. Sherry, K. Schaefer, and R. Duits.
          "Diffusion-Shock Filtering on the Space of Positions and
          Orientations." In: Scale Space and Variational Methods in Computer
          Vision (2025), pp. .
          DOI:.
    """
    τ_D = compute_timestep_diffusion(dxy, dθ, G_D_inv)
    τ_M = compute_timestep_shock(dxy, dθ, G_S_inv)
    return min(τ_D, τ_M)

def compute_timestep_diffusion(dxy, dθ, G_D_inv):
    """
    Compute timestep to solve Diffusion PDE, such that the scheme retains the
    maximum-minimum principle of the continuous PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_D_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the diffusion.
    
    Returns:
        timestep, taking values greater than 0.
    """
    return 1 / (4 * ((G_D_inv[0] + G_D_inv[1]) / dxy**2 + G_D_inv[2] / dθ**2))

def compute_timestep_shock(dxy, dθ, G_S_inv):
    """
    Compute timestep to solve (Coherence-Enhancing) Shock PDE, such that the
    scheme retains the maximum-minimum principle of the continuous PDE.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_S_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the shock.
    
    Returns:
        timestep, taking values greater than 0.
    """
    return 1 / (np.sqrt((G_S_inv[0] + G_S_inv[1]) / dxy**2 + G_S_inv[2] / dθ**2))


# TR-TV Flow

def TV_enhancing(u0_np_unscaled, ground_truth_np, G_inv_np, dxy, dθ, θs_np, σ_s, σ_o, T, dt=None, λ=1.):
    """
    Perform left-invariant Total Roto-Translational Variation (TR-TV) Flow in
    M_2.[1][2]

    Args:
        `u0_np`: np.ndarray initial condition, with shape [Nx, Ny, Nθ].
        `mask_np`: np.ndarray inpainting mask, with shape [Nx, Ny, Nθ], taking
          values 0 and 1. Wherever the value is 1, no inpainting happens.
        `θs_np`: np.ndarray orientation coordinate θ throughout the domain.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis.
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
        `σ_s`: standard deviation of the internal regularisation in the spatial
          direction of the perpendicular laplacian, used for determining whether
          to perform dilation or erosion.
        `σ_o`: standard deviation of the internal regularisation in the
          orientational direction of the perpendicular laplacian.
        `T`: time that image is evolved under the DS PDE.

    Returns:
        np.ndarray solution to the TV Flow PDE with initial condition `u0_np` at
        time `T`.

    References:
        [1]: A. Chambolle and Th. Pock.
          "Total roto-translational variation." In: Numerische Mathematik
          (2019), pp. 611--666.
          DOI:10.1137/s00211-019-01026-w.
        [2]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
          "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
          Positions and Orientations." In: Journal of Mathematical Imaging and
          Vision (2021).
          DOI:10.1007/s10851-020-00991-4.
    """
    if dt is None:
        dt = compute_timestep_TV(dxy, dθ, G_inv_np)
    print(dt)
    n = int(T / dt)
    k_s, radius_s = gaussian_kernel(σ_s)
    k_o, radius_o = gaussian_kernel(σ_o)
    u0_np = u0_np_unscaled * λ
    shape = u0_np.shape
    u = ti.field(dtype=ti.f32, shape=shape)
    u.from_numpy(u0_np)
    G_inv = ti.Matrix(G_inv_np, dt=ti.f32)
    θs = ti.field(dtype=ti.f32, shape=shape)
    θs.from_numpy(θs_np)
    A1_u = ti.field(dtype=ti.f32, shape=shape)
    A2_u = ti.field(dtype=ti.f32, shape=shape)
    A3_u = ti.field(dtype=ti.f32, shape=shape)
    grad_norm_u = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_1 = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_2 = ti.field(dtype=ti.f32, shape=shape)
    normalised_grad_3 = ti.field(dtype=ti.f32, shape=shape)
    TV_u = ti.field(dtype=ti.f32, shape=shape)
    storage = ti.field(dtype=ti.f32, shape=shape)

    ## Image Quality Measures
    max_val = 255. # Images are assumed to take gray values in [0, 255].
    ground_truth = ti.field(dtype=ti.f32, shape=shape[:-1])
    ground_truth.from_numpy(ground_truth_np)
    print(ground_truth.shape)
    u_projected = ti.field(dtype=ti.f32, shape=shape[:-1])
    project_down(u, u_projected, 0., max_val, λ)
    PSNR = [compute_PSNR(u_projected, ground_truth, max_val)]
    L2 = [compute_L2(u_projected, ground_truth)]
    L1 = [compute_L1(u_projected, ground_truth)]

    for _ in tqdm(range(n)):
        TV(u, G_inv, dxy, dθ, θs, k_s, radius_s, k_o, radius_o, A1_u, A2_u, A3_u, grad_norm_u, normalised_grad_1,
           normalised_grad_2, normalised_grad_3, TV_u, storage)
        step_TV(u, dt, TV_u)

        project_down(u, u_projected, 0., max_val, λ)
        PSNR.append(compute_PSNR(u_projected, ground_truth, max_val))
        L2.append(compute_L2(u_projected, ground_truth))
        L1.append(compute_L1(u_projected, ground_truth))
    return u.to_numpy() / λ, np.array(PSNR), np.array(L2), np.array(L1)
    
@ti.kernel
def step_TV(
    u: ti.template(),
    dt: ti.f32,
    TV_u: ti.template(),
):
    """
    @taichi.kernel

    Perform a single timestep of TR-TV flow.

    Args:
      Static:
        `dt`: step size, taking values greater than 0.
        `TV_u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) of 
          div(grad `u` / ||grad `u`||), which is updated in place.
      Mutated:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) which we want to evolve
          with the shock PDE.
    """
    for I in ti.grouped(TV_u):
        u[I] += dt * TV_u[I]

def compute_timestep_TV(dxy, dθ, G_inv):
    """
    Compute timestep to solve TR-TV flow.
    
    Args:
        `dxy`: step size in x and y direction, taking values greater than 0.
        `dθ`: step size in θ direction, taking values greater than 0.
        `G_inv_np`: np.ndarray(shape=(3,), dtype=[float]) of constants of the
          inverse of the diagonal metric tensor with respect to left invariant
          basis used to define the TR-TV flow.
    
    Returns:
        timestep, taking values greater than 0.
    """
    return dxy**2 * dθ / (2 * ((G_inv[0] + G_inv[1]) * dxy + G_inv[2] * dθ))

# Fix padding function

@ti.kernel
def fill_u_switch(
    u: ti.template(),
    u_switch: ti.template()
):
    """
    @taichi.kernel

    Update the content of the field used to determine the switch.

    Args:
      Static:
        `u`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) content to fill
          `padded_u`.
      Mutated:
        `u_switch`: ti.field(dtype=[float], shape=[Nx, Ny, Nθ]) storage array,
          updated in place.
    """
    for I in ti.grouped(u_switch):
        u_switch[I] = u[I]