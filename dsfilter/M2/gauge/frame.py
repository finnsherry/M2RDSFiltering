"""
    frame
    =====

    Compute the gauge frames fitted to data on M_2, using:
      1. `compute_gauge_frame_and_orientation_confidence`: fit the gauge frame
      and compute the orientation confidence, see [2].
      2. `compute_deviation_from_horizontality`: compute the deviation from
      horizontality given the main gauge vector.
      3. `compute_curvature`: compute the curvature given the main gauge vector.

    References:
      [1]: R. Duits, B.M.N. Smets, A.J. Wemmenhove, J.W. Portegies, and
      E. Bekkers.
      "Recent Geometric Flows in Multi-Orientation Image Processing via a Cartan
      Connection." In: Handbook of Mathematical Models and Algorithms in
      Computer Vision and Imaging: Mathematical Imaging and Vision (2021),
      pp. 1--60.
      DOI:10.1007/978-3-030-98661-2_101.
      [2]: E.M. Franken. Supervised by B.M. ter Haar Romeny and R. Duits.
      "Regularized Derivatives and Local Features", Ch. 5 in "Enhancement of
      Crossing Elongated Structures in Images" (2008), pp. 111--132.
      ISBN:978-90-384-1456-4.
"""

import numpy as np
import scipy as sp

# NumPy implementation

def compute_gauge_frame_and_orientation_confidence(U, dxy, dθ, θs, ξ, ρ_s=1., ρ_o=0.5):
    """
    Compute gauge frames, with respect to the left invariant frame, to fit the
    data `U`, as well as the orientation confidence.[1][2, Section 5.4]

    Args:
        `U`: np.ndarray orientation score, with shape [Nx, Ny, Nθ].
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
        `θs`: np.ndarray orientation coordinate θ throughout the domain.
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
      Optional:
        `ρ_s`: standard deviation in pixels of the spatial external
          regularisation, taking values greater than 0. Defaults to 1.
        `ρ_o`: standard deviation in pixels of the orientational external
          regularisation, taking values greater than 0. Defaults to 0.5.

    Returns:
        Tuple of the vector fields B_1, B_2, and B_3 defining the gauge frames,
        with respect to the left invariant frame, and the orientation
        confidence.
    
    References:
        [1]: R. Duits, B.M.N. Smets, A.J. Wemmenhove, J.W. Portegies, and
          E. Bekkers.
          "Recent Geometric Flows in Multi-Orientation Image Processing via a
          Cartan Connection." In: Handbook of Mathematical Models and Algorithms
          in Computer Vision and Imaging: Mathematical Imaging and Vision
          (2021), pp. 1--60.
          DOI:10.1007/978-3-030-98661-2_101.
        [2]: E.M. Franken. Supervised by B.M. ter Haar Romeny and R. Duits.
          "Regularized Derivatives and Local Features," Ch. 5 in "Enhancement of
          Crossing Elongated Structures in Images" (2008), pp. 111--132.
          ISBN:978-90-384-1456-4.
    """
    # "Hessian" matrix induced by the Lie-Cartan 0 connection, which is defined
    # as the affine connection such that
    #   ∇_A_i A_j = 0. 
    H = compute_Hessian(U, dxy, dθ, θs)
    B1, B2, B3 = compute_gauge_frame(H, ξ, ρ_s=ρ_s, ρ_o=ρ_o)
    oc = compute_orientation_confidence(H, B2, B3)
    return B1, B2, B3, oc

def compute_orientation_confidence(H, B2, B3):
    """
    Compute the orientation confidence[2] of an orientation score from its
    Lie-Cartan 0 induced Hessian `H`[1] and the gauge vectors perpendicular to
    the local orientation `B2` and `B3`.

    Args:
        `H`: np.ndarray Hessian matrix of orientation score, with shape
          [Nx, Ny, Nθ, 3, 3].
        `Bi`: np.ndarray i-th gauge vector with respect to the left invariant
          frame, with shape [Nx, Ny, Nθ, 3]

    Returns:
        Orientation confidence.

    
    References:
        [1]: R. Duits, B.M.N. Smets, A.J. Wemmenhove, J.W. Portegies, and
          E. Bekkers.
          "Recent Geometric Flows in Multi-Orientation Image Processing via a
          Cartan Connection." In: Handbook of Mathematical Models and Algorithms
          in Computer Vision and Imaging: Mathematical Imaging and Vision
          (2021), pp. 1--60.
          DOI:10.1007/978-3-030-98661-2_101.
        [2]: E.M. Franken. Supervised by B.M. ter Haar Romeny and R. Duits.
          "Regularized Derivatives and Local Features," Ch. 5 in "Enhancement of
          Crossing Elongated Structures in Images" (2008), pp. 111--132.
          ISBN:978-90-384-1456-4.
    """
    # The orientation confidence is given by
    #   oc = -Δ_perp U = -(B_2^2 + B_3^2) U.
    # In the left invariant frame, this may be computed using
    #   B_i^2 U = B_i^T H B_i,
    # so that
    #   oc = - B_2^T H B_2 - B_3^T H B_3.
    return -(
        B2[..., None, :] @ H @ B2[..., None] +
        B3[..., None, :] @ H @ B3[..., None]
    )[..., 0, 0]

def compute_curvature(B1):
    """
    Compute the curvature given principal gauge vector `B1` with respect to the
    left invariant frame.
    """
    c1 = B1[..., 0]
    c2 = B1[..., 1]
    c3 = B1[..., 2]
    # 🤔 Is it strange that there is no ξ in this expression?
    return c3 * np.sign(c1) / np.sqrt(c1**2 + c2**2)

def compute_deviation_from_horizontality(B1):
    """
    Compute the deviation from horizontality given principal gauge vector `B1`
    with respect to the left invariant frame.
    """
    c1 = B1[..., 0]
    c2 = B1[..., 1]
    return np.arctan2(c2, c1)

def compute_gauge_frame(H, ξ, ρ_s=1., ρ_o=0.5):
    """
    Compute gauge frames, with respect to the left invariant frame, to fit the
    data of which `H` is the Lie-Cartan 0 induced Hessian.[1]

    Args:
        `U`: np.ndarray orientation score, with shape [Nx, Ny, Nθ].
        `ξ`: stiffness parameter defining the cost of moving one unit in the
          orientatonal direction relative to moving one unit in a spatial
          direction, taking values greater than 0.
      Optional:
        `ρ_s`: standard deviation in pixels of the spatial external
          regularisation, taking values greater than 0. Defaults to 1.
        `ρ_o`: standard deviation in pixels of the orientational external
          regularisation, taking values greater than 0. Defaults to 0.5.

    Returns:
        Tuple of the vector fields B_1, B_2, and B_3 defining the gauge frames,
        with respect to the left invariant frame.
    
    References:
        [1]: R. Duits, B.M.N. Smets, A.J. Wemmenhove, J.W. Portegies, and
          E. Bekkers.
          "Recent Geometric Flows in Multi-Orientation Image Processing via a
          Cartan Connection." In: Handbook of Mathematical Models and Algorithms
          in Computer Vision and Imaging: Mathematical Imaging and Vision
          (2021), pp. 1--60.
          DOI:10.1007/978-3-030-98661-2_101.
    """
    H_T = np.moveaxis(H, -1, -2)
    # Metric defining the relation between the spatial dimensions and the
    # orientational directions.
    M_inv = np.diag((1./ξ, 1./ξ, 1.))
    # Matrix equation formulation of minimisation problem.
    A = M_inv @ H_T @ M_inv @ M_inv @ H @ M_inv
    # Regularise spatially.
    A_reg = sp.ndimage.gaussian_filter(A, (ρ_s, ρ_s, ρ_o), axes=range(A.ndim-2), mode="wrap")

    # The eigenvector corresponding to the smallest eigenvalue of this matrix
    # is closely related to the minimiser we are looking for: if v is that
    # eigenvector, then the minimiser c we are looking for is given by
    #   c = M_inv v.
    # Additionally, if v has unit (Euclidean) length, then c is properly
    # normalised.
    _, vs = np.linalg.eigh(A_reg)
    c = vs[..., 0] @ M_inv

    # The direction of c is not fixed by the minimisation problem; we choose
    # B_1 such that it points "in the same direction" as A_1.
    B1 = c * np.sign(c[..., 0])[..., None]
    c1 = B1[..., 0]
    c2 = B1[..., 1]
    c3 = B1[..., 2]
    # Spatial angle between B_1 and A_1, also called the deviation from
    # horizontality.
    χ = np.arctan2(c2, c1)
    # Orientational angle between B_1 and the plane spanned by A_1 and A_2,
    # related to the dimensionless curvature by
    #   κ / ξ = tan(ν).
    ν = np.arctan2(c3, ξ * np.sqrt(c1**2 + c2**2))
    # We further impose that the frame is orthogonal with respect to
    #   G = diag(ξ^2, ξ^2, 1).
    # We choose B_2 to be purely spatial. Finally, we want the frame to be
    # right-handed.
    B2 = np.zeros_like(B1)
    B2[..., 0] = -np.sin(χ) / ξ
    B2[..., 1] = np.cos(χ) / ξ
    B3 = np.zeros_like(B1)
    B3[..., 0] = -np.sin(ν) * np.cos(χ) / ξ
    B3[..., 1] = -np.sin(ν) * np.sin(χ) / ξ
    B3[..., 2] = np.cos(ν)
    return B1, B2, B3

def compute_Hessian(U, dxy, dθ, θs, σ=1.):
    """
    Compute the Lie-Cartan 0 induced Hessian of `U`.[1]
    The Hessian matrix is given by
        H_j^i := A_j A_i `U`,
    where the upper index i gives the row, and the lower index j gives the
    column.

    Args:
        `U`: np.ndarray orientation score, with shape [Nx, Ny, Nθ].
        `dxy`: size of pixels in the x- and y-directions.
        `dθ`: size of pixels in the θ-direction.
        `θs`: np.ndarray orientation coordinate θ throughout the domain.

    Returns:
        np.ndarray Hessian matrix field corresponding to `U`, with shape
        [Nx, Ny, Nθ, 3, 3].
    
    References:
        [1]: R. Duits, B.M.N. Smets, A.J. Wemmenhove, J.W. Portegies, and
          E. Bekkers.
          "Recent Geometric Flows in Multi-Orientation Image Processing via a
          Cartan Connection." In: Handbook of Mathematical Models and Algorithms
          in Computer Vision and Imaging: Mathematical Imaging and Vision
          (2021), pp. 1--60.
          DOI:10.1007/978-3-030-98661-2_101.

    Notes:
        The Lie-Cartan 0 connection is defined as the affine connection with
            ∇_A_i A_j = 0.
        In [1], the authors use a different ordering of the frame vectors: what
        we call A_1 and B_1 they call A_2 and B_2. The frames are left handed in
        both cases, so our A_2 and B_2 corresponds to -A_1 and -B_1 in [1].
    """
    shape = U.shape
    cos = np.cos(θs)
    sin = np.sin(θs)

    # First order derivatives.
    dx_U = x_derivative(U, dxy)
    dy_U = y_derivative(U, dxy)
    dθ_U = θ_derivative(U, dθ)
    A1_U = cos * dx_U + sin * dy_U
    A2_U = -sin * dx_U + cos * dy_U
    A3_U = dθ_U

    # Second order derivatives.
    # Structure of Arrays ordering.
    H = np.zeros((3, 3, *shape))
    dx_A1_U = x_derivative(A1_U, dxy)
    dy_A1_U = y_derivative(A1_U, dxy)
    dθ_A1_U = θ_derivative(A1_U, dθ)
    H[0, 0] = cos * dx_A1_U + sin * dy_A1_U
    H[0, 1] = -sin * dx_A1_U + cos * dy_A1_U
    H[0, 2] = dθ_A1_U
    dx_A2_U = x_derivative(A2_U, dxy)
    dy_A2_U = y_derivative(A2_U, dxy)
    dθ_A2_U = θ_derivative(A2_U, dθ)
    H[1, 0] = cos * dx_A2_U + sin * dy_A2_U
    H[1, 1] = -sin * dx_A2_U + cos * dy_A2_U
    H[1, 2] = dθ_A2_U
    dx_A3_U = x_derivative(A3_U, dxy)
    dy_A3_U = y_derivative(A3_U, dxy)
    dθ_A3_U = θ_derivative(A3_U, dθ)
    H[2, 0] = cos * dx_A3_U + sin * dy_A3_U
    H[2, 1] = -sin * dx_A3_U + cos * dy_A3_U
    H[2, 2] = dθ_A3_U

    # Array of Structures ordering.
    axes = range(H.ndim)
    return H.transpose((*axes[2:], *axes[:2]))

def x_derivative(U, dxy):
    return np.gradient(U, dxy, axis=0)

def y_derivative(U, dxy):
    return np.gradient(U, dxy, axis=1)

def θ_derivative(U, dθ):
    dθ_U = np.zeros_like(U)
    dθ_U[..., 1:-1] = U[..., 2:] - U[..., :-2]
    dθ_U[..., 0] = U[..., 1] - U[..., -1]
    dθ_U[..., -1] = U[..., 0] - U[..., -2]
    dθ_U /= (2 * dθ)
    return dθ_U