"""
analysis

Tools to analyze an evolution from the data
"""
import numpy as np
from evolver.eoms import N_efolds
from evolver.initialize import unpack

def analyze(a, epsilon):
    """Analyze an evolution"""
    results = {}

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

def load_data(filename):
    """Loads data from a run"""
    with open(filename + ".dat") as f:
        data = f.readlines()
    with open(filename + ".dat2") as f:
        data2 = f.readlines()
    with open(filename + ".info") as f:
        data3 = f.readlines()

    # Process the data
    # Sum the number of l=0 and l=1 modes
    nummodes = int(data3[1].split(":")[1]) + int(data3[2].split(":")[1])
    # Read the numbers from the raw data
    results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
    results2 = np.array([list(map(float, line.split(", "))) for line in data2]).transpose()
    # Unpack into nicer variables
    t = results[0]
    a, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpack(results[1:], nummodes)
    (H, Hdot, addot, phi0ddot, phi2pt, phi2ptdt, phi2ptgrad, rho,
        deltarho2, epsilon, V, Vd, Vdd, Vddd, Vdddd) = results2[1:]

    # Put the data into a container
    fulldata = {
        "t": t,
        "a": a,
        "adot": a * H,
        "phi0": phi0,
        "phi0dot": phi0dot,
        "phiA": phiA,
        "phidotA": phidotA,
        "psiA": psiA,
        "phiB": phiB,
        "phidotB": phidotB,
        "psiB": psiB,
        "H": H,
        "Hdot": Hdot,
        "addot": addot,
        "phi0ddot": phi0ddot,
        "phi2pt": phi2pt,
        "phi2ptdt": phi2ptdt,
        "phi2ptgrad": phi2ptgrad,
        "rho": rho,
        "deltarho2": deltarho2,
        "epsilon": epsilon,
        "V": V,
        "Vd": Vd,
        "Vdd": Vdd,
        "Vddd": Vddd,
        "Vdddd": Vdddd,
    }

    # Return __everything__
    return fulldata
