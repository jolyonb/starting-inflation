import matplotlib.pyplot as plt
import numpy as np
import argparse
import eoms
from run import params
from initialize import unpack
from math import exp
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from run import lamda
from run import Rmax
from run import timestep

# Read in the data
with open(params.filename) as f:
    data = f.readlines()
with open(params.filename2) as f:
    data2 = f.readlines()

# Process the data
results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
results2 = np.array([list(map(float, line.split(", "))) for line in data2]).transpose()
# Unpack the data
t = results[0]
a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpack(results[1:], params.total_wavenumbers)

(H, Hdot, addot, phi0ddot, hpotential0, hgradient0, hkinetic0, rho, 
    deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd) = results2[1:]

phi = [None] * params.k_modes
phidot = [None] * params.k_modes
psi = [None] * params.k_modes
# Attach coefficients
for i in range(params.k_modes):
    params.poscoeffs[0][i]
    phi[i] = params.poscoeffs[0][i] * phiA[i] + params.velcoeffs[0][i] * phiB[i]
    phidot[i] = params.poscoeffs[0][i] * phidotA[i] + params.velcoeffs[0][i] * phidotB[i]
    psi[i] = params.poscoeffs[0][i] * psiA[i] + params.velcoeffs[0][i] * psiB[i]

def eps_function():
	return (-Hdot/(H*H))

blackout = 10
factoid = len(eps_function())
eps_times =[]
eps_values = []
for i in range(blackout,factoid):
	if 0.95 <= eps_function()[i] <= 1.05:
		eps_times.append(i)
		eps_values.append(np.abs(eps_function()[i]-1.0))

infend_index = eps_values.index(min(eps_values))
infend_timestamp = eps_times[infend_index]
infend_physical_time = infend_timestamp * timestep
