"""
analysis

Tools to analyze an evolution from the data
"""
import numpy as np
from evolver.eoms import slow_roll_epsilon, N_efolds

def analyze(a, adot, addot):
    """Analyze an evolution"""
    results = {}

    # First, compute epsilon
    epsilon = slow_roll_epsilon(a, adot, addot)

    # Find the minimum epsilon
    min_eps = np.min(epsilon)

    # Did we ever hit slow roll?
    results["slowroll"] = min_eps < 0.1
    if not results["slowroll"]:
        return results

    # Terminated once epsilon > 1?
    results["inflationended"] = epsilon[-1] >= 1

    # Number of efolds passed
    results["efolds"] = N_efolds(a[-1])

    return results
