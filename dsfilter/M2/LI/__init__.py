"""
    LI
    ==

    Apply left-invariant Diffusion-Shock filtering on M_2.

    Provides the following "top level" submodule:
      1. filter: perform left-invariant RDS filtering on M_2.

    Additionally, we have the following "internal" submodules
      1. derivatives: compute various derivatives of functions on M_2.
      2. switches: compute the quantities that switch between diffusion and
      shock and between erosion and dilation.
"""

# Access entire backend
import dsfilter.M2.LI.filter
import dsfilter.M2.LI.derivatives
import dsfilter.M2.LI.switches