"""
    M2
    ==

    Apply Diffusion-Shock filtering on M_2.

    Provides the following "top level" submodule:
      1. LI: perform DS filtering on M_2 using left invariant vector fields.
      2. gauge: perform DS filtering on M_2 using gauge frames.

    Additionally, we have the following "internal" submodules
      1. regularisers: 
      2. utils: 
"""

# Access entire backend
import dsfilter.M2.regularisers
import dsfilter.M2.utils
import dsfilter.M2.LI
import dsfilter.M2.gauge