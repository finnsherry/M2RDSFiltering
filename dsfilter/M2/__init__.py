"""
    M2
    ==

    Apply Diffusion-Shock filtering on M_2.

    Provides the following "top level" submodule:
      1. LI: perform left-invariant RDS filtering on M_2.
      2. gauge: perform gauge frame RDS filtering on M_2.

    Additionally, we have the following "internal" submodules
      1. regularisers: regularise functions on M_2 with Gaussian filters.
      2. utils: miscellaneous utilities.
"""

# Access entire backend
import dsfilter.M2.regularisers
import dsfilter.M2.utils
import dsfilter.M2.LI
import dsfilter.M2.gauge