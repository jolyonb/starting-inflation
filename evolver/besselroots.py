# -*- coding: utf-8 -*-
"""
besselroots.py

Computes roots of spherical Bessel functions recursively
"""
from math import pi
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq

def get_jn_roots(ell_max, num_modes):
    """
    Compute the roots of spherical bessel functions j_ell(r) up to ell_max
    For ell = 0, compute num_modes modes
    For ell > 0, make sure all roots are less than the largest root for ell = 0

    Arguments:
        * ell_max: The maximum ell to consider
        * num_modes: The number of modes to compute for ell = 0

    Returns:
        * A list of numpy arrays. The list will have at most ell_max entries, indexed
          by ell. Each numpy array contains the roots of j_ell(r). No empty arrays are
          returned, so the actual maximum ell may be less than ell_max.
    """
    # Because of the bracketing technique we use to obtain zeroes for higher ell modes,
    # we need to go a bit further out to make sure that we get all of the zeroes for the
    # higher ell values. We make sure that we will have num_k_modes zeroes for the
    # ell_max mode. We will truncate later.
    root_vals = [None for i in range(ell_max + 1)]
    root_vals[0] = np.array([n * pi for n in range(1, num_modes + ell_max + 1)])
    max_root = root_vals[0][num_modes - 1]  # Stores the maximum root that we will end up keeping

    # We need to wrap the spherical_jn call to reorder the arguments
    def j_n(r, n):
        return spherical_jn(n, r)

    # The roots of j_(n+1) interlace the roots of j_n
    # This gives us bounds to perform a bracketed root search
    for ell in range(1, ell_max + 1):
        roots = [None for i in range(len(root_vals[ell - 1]) - 1)]
        for root in range(len(root_vals[ell - 1]) - 1):
            roots[root] = brentq(j_n,
                                 root_vals[ell - 1][root],
                                 root_vals[ell - 1][root + 1],
                                 (ell,))
        root_vals[ell] = np.array(roots)

    # Now we perform the truncation
    for ell in range(0, ell_max + 1):
        # Find the first element that is > max_root
        element = -1
        for element in range(len(root_vals[ell])):
            if root_vals[ell][element] > max_root:
                break
        else:
            continue
        if element > 0:
            # Truncate at this element
            root_vals[ell] = root_vals[ell][0:element]
        else:
            # No more roots
            root_vals = root_vals[0:ell]
            break

    return root_vals

if __name__ == "__main__":
    # Just run some quick tests
    print("Running consistency checks")

    # Set these appropriately
    ell_max = 100
    num_modes = 20

    # Get the roots
    root_vals = get_jn_roots(ell_max, num_modes)

    # What is our actual ell_max?
    if len(root_vals) < ell_max + 1:
        ell_max = len(root_vals) - 1
        print("Lowering ell_max to {}".format(ell_max))

    # Make sure we have the right number of modes
    assert len(root_vals[0]) == num_modes

    # Test that our roots are actually roots
    for ell in range(0, ell_max + 1):
        assert np.all(np.abs(spherical_jn(ell, root_vals[ell])) < 1e-12)

    # Make sure all roots are smaller than the maximum root set by ell = 0
    max_root = root_vals[0][-1]
    for ell in range(0, ell_max + 1):
        assert np.all(root_vals[ell] <= max_root)

    # Check that the spherical bessel functions at the max root have the same
    # sign as just after the last root that we found for them in the range
    epsilon = 0.1
    for ell in range(1, ell_max + 1):
        root_val = spherical_jn(ell, root_vals[ell][-1] + epsilon)
        root_max_val = spherical_jn(ell, max_root)
        assert np.sign(root_val) == np.sign(root_max_val)

    print("All checks passed!")
