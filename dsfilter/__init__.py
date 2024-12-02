"""
    DSFilter
    ======

    The Python package *dsfilter* contains methods to apply Diffusion-Shock
    filtering, as described in Schaefer and Weickert,[1][2] on R^2 and M_2. The
    primary methods are:
      1. `DS_enhancing_LI`: perform left-invariant RDS filtering on M_2 for
      denoising.
      2. `DS_enhancing_gauge`: perform gauge frame RDS filtering on M_2 for
      denoising.
      3. `DS_enhancing_LI`: perform RDS filtering on R^2 for denoising.
      4. `TV_enhancing_LI`: perform left-invariant TR-TV flow on M_2 for
      denoising.[3][4]
      5. `TV_enhancing_gauge`: perform gauge frame TR-TV flow on M_2 for
      denoising.[3][4]
      6. `DS_inpainting_LI`: perform left-invariant RDS inpainting on M_2.
      7. `DS_inpainting_gauge`: perform gauge frame RDS inpainting on M_2.

    Summary: enhance images by applying Diffusion-Shock filtering in R^2 and
    M_2.
    
    References:
      [1]: K. Schaefer and J. Weickert.
      "Diffusion-Shock Inpainting." In: Scale Space and Variational Methods in
      Computer Vision 14009 (2023), pp. 588--600.
      DOI:10.1137/15M1018460.
      [2]: K. Schaefer and J. Weickert.
      "Regularised Diffusion-Shock Inpainting." In: Journal of Mathematical
      Imaging and Vision (2024).
      DOI:10.1007/s10851-024-01175-0.
      [3]: K. Schaefer and J. Weickert.
      "Total roto-translational variation." In: Numerische Mathematik (2019),
      pp. 611--666.
      DOI:10.1137/s00211-019-01026-w.
      [4]: B.M.N. Smets, J.W. Portegies, E. St-Onge, and R. Duits.
      "Total Variation and Mean Curvature PDEs on the Homogeneous Space of
      Positions and Orientations." In: Journal of Mathematical Imaging and
      Vision (2021).
      DOI:10.1007/s10851-020-00991-4.
"""

# Access entire backend
import dsfilter.utils
import dsfilter.visualisations
import dsfilter.orientationscore
import dsfilter.R2
import dsfilter.M2

# Most important functions are available at top level
## R2
from dsfilter.R2.filter import DS_inpainting as DS_inpainting_R2
from dsfilter.R2.filter import DS_enhancing as DS_enhancing_R2
## M2
### Left invariant
from dsfilter.M2.LI.filter import DS_inpainting as DS_inpainting_LI
from dsfilter.M2.LI.filter import DS_enhancing as DS_enhancing_LI
from dsfilter.M2.LI.filter import TV_enhancing as TV_enhancing_LI
### Gauge
from dsfilter.M2.gauge.filter import DS_enhancing as DS_enhancing_gauge
from dsfilter.M2.gauge.filter import TV_enhancing as TV_enhancing_gauge