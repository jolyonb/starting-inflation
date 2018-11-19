#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a quick and dirty plot of data
"""
import matplotlib.pyplot as plt
import numpy as np
import argparse
import eoms
from run import params
from initialize import unpack
from math import exp

#
# Deal with command line arguments
#
parser = argparse.ArgumentParser(description="Plot data from a given data file using column 1 on the x axis")

def pos_int(value):
    """Helper function used to select a positive integer"""
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError("must be a positive integer. You supplied: %s" % value)
    return int(value)

# Which type of plot to make
parser.add_argument("-l", help="Log plot type (0=None, 1=y, 2=x, 3=both)",
                    default=0, type=pos_int, dest="logplot")

# Parse the command line
args = parser.parse_args()


# Read in the data
with open(params.filename) as f:
    data = f.readlines()
with open(params.filename2) as f:
    data2 = f.readlines()

# Process the data
results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
results2 = np.array([list(map(float, line.split(", "))) for line in data2]).transpose()

def make_plot(xvals, ylistvals, plottype, numsteps):
    """Create a simple plot of xvals and yvals"""
    fig, ax = plt.subplots()

    # Figure out the plot type
    if plottype == 0:
        plotter = ax.plot
    elif plottype == 1:
        plotter = ax.semilogy
    elif plottype == 2:
        plotter = ax.semilogx
    elif plottype == 3:
        plotter = ax.loglog
    else:
        print("{} is not a valid Log plot type".format(plottype))
        return

    for entry in ylistvals:
        plotter(xvals[0:numsteps], entry[0:numsteps])
    ax.set(xlabel='time', ylabel='column')
    ax.grid()
    plt.show()

# Unpack the data
t = results[0]
a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpack(results[1:], params.total_wavenumbers)

(H, Hdot, addot, phi0ddot, hpotential0, hgradient0,
 hkinetic0, rho, deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd) = results2[1:]

phi = [None] * params.k_modes
phidot = [None] * params.k_modes
psi = [None] * params.k_modes
# Attach coefficients
for i in range(params.k_modes):
    phi[i] = params.poscoeffs[0][i] * phiA[i] + params.velcoeffs[0][i] * phiB[i]
    phidot[i] = params.poscoeffs[0][i] * phidotA[i] + params.velcoeffs[0][i] * phidotB[i]
    psi[i] = params.poscoeffs[0][i] * psiA[i] + params.velcoeffs[0][i] * psiB[i]

plotit = [np.abs(mode) for mode in phi]
#plotit = [adot/a]

#plotit = [deltarho2 / rho]

#plotit = [hpotential0 / rho * eoms.ddpotential(phi0, params)]

#plotit = [hpotential0, hgradient0, hkinetic0]

# Create the plot
make_plot(t, plotit, args.logplot, -1)
