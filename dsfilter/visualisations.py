"""
    visualisations
    ==============
    
    Provides methods to visualise 2D and 3D images using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt

def align_to_standard_array_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to arrays aligned with real
    axes, with respect to standard array convention (see Notes for more 
    explanation).

    Args:
        `field`: np.ndarray of scalar field given in indices with respect to
          arrays aligned with real axes.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
           real axes aligned                 standard
            I x ------                    I ^ ------
            | | |    |        =>          | | |    |
            v v ------                    v y ------
                 y ->                          x ->
                 J ->                          J ->  
    """
    # field_transposed = np.transpose(field, axes=(1, 0, 2))
    field_transposed = field.swapaxes(1, 0)
    field_aligned = np.flip(field_transposed, axis=0)
    return field_aligned

def align_to_real_axis_scalar_field(field):
    """
    Align `field`, given in indices with respect to standard array convention, 
    with real axes (see Notes for more explanation).

    Args:
        `field`: np.ndarray of scalar field given with respect to standard array
          convention.

    Notes:
        By default, if you take a point in an image, and want to move a single
        pixel up, you do so by decreasing I, while if you want to move a single
        pixel to the right, you do so by increasing J. Hence, the shape of the
        array is [Ny, Nx, Nθ].

        When aligned with real axes, moving up a single pixel is achieved by
        increasing J, and moving right a single pixel is achieved by increasing
        I. Hence, the shape of the array is [Nx, Ny, Nθ].

        Alignment is achieved by first flipping and subsequently transposing the
        array.
            
    ===================== DRAWING DOES NOT WORK IN HELP ========================    
        
               standard                  real axes aligned
            I ^ ------                    I x ------
            | | |    |        =>          | | |    |
            v y ------                    v v ------
                 x ->                          y ->
                 J ->                          J ->  
    """
    field_flipped = np.flip(field, axis=0)
    field_aligned = field_flipped.swapaxes(1, 0)
    return field_aligned

def plot_image_array(image_array, x_min, x_max, y_min, y_max, cmap="gray", figsize=(10, 10), clip=None,
                     rasterized=False, fig=None, ax=None):
    """Plot `image_array` as a heatmap."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if clip is not None:
        image_array = np.clip(image_array, clip[0], clip[1])

    image_array_aligned = align_to_standard_array_axis_scalar_field(image_array)

    cbar = ax.imshow(image_array_aligned, cmap=cmap, extent=(x_min, x_max, y_min, y_max), rasterized=rasterized)
    return fig, ax, cbar