#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
import pickle
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from evolver.model import Model
from evolver.utilities import load_data, analyze

####################################
# Deal with command line arguments #
####################################
parser = argparse.ArgumentParser(description="Plot data from a sweep")
parser.add_argument("filename", help="Base of the output name to read data in from")
# parser.add_argument("outfilename", help="Filename to output to (include .pdf)")
args = parser.parse_args()

# Specify the critical number of efolds above/below which we determine
# sufficient/insufficient inflation
Nef_crit = 65.0

# Select what to plot
plot_types = {"off": True, "bunchdavies": False, "hartree": False}

def plot3d(phi0, phi0dot, value, name):
    fig = plt.figure(figsize=(7.0, 7.0), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Grab the phi0 values
    phi0vals = sorted(list(set(phi0)))
    phi0dotvals = sorted(list(set(phi0dot)))
    phi0mesh, phi0dotmesh = np.meshgrid(phi0vals, phi0dotvals)
    zmesh = np.zeros_like(phi0mesh)

    for idx in range(len(value)):
        xidx = phi0vals.index(phi0[idx])
        yidx = phi0dotvals.index(phi0dot[idx])
        zmesh[xidx, yidx] = value[idx]

    # Plot the data points (we could reconstruct a surface if we desired)
    # ax.scatter(phi0, phi0dot, value)
    ax.plot_wireframe(phi0mesh, phi0dotmesh, zmesh)

    # Plot the critical threshold
    xmin = phi0[0]
    xmax = phi0[-1]
    ymin = phi0dot[0]
    ymax = phi0dot[-1]

    X, Y = np.meshgrid([xmin, xmax], [ymin, ymax])
    Z = 0*X + Nef_crit
    ax.plot_surface(X, Y, Z, alpha=0.2)

    # Add some labels
    ax.set_xlabel(r'$\phi_0$')
    ax.set_ylabel(r'$\dot{\phi}_0$')
    ax.set_zlabel(name)

    return fig

# Parse the filename
filename = args.filename
directory, filename = os.path.split(filename)
olddir = os.getcwd()
os.chdir(directory)

# Suck up the data
with open(filename + "-info.txt") as f:
    lines = f.readlines()

# Find all of the runs to read from
data = []
for line in lines[1:]:
    if line:
        fn, phi0, phi0dot = line.strip().split("\t")
        phi0 = float(phi0)
        phi0dot = float(phi0dot)
        data.append((fn, phi0, phi0dot))

# Read the data files for each run in the sweep
plot_data = {
  "phi0": [],
  "phi0dot": [],
  "H": [],
  "rho": [],
  "deltarho2": [],
  "phi2pt": [],
  "efolds": [],
  "infl": [],
  "kappa": [],
}

for file, phi0, phi0dot in data:
    if file.endswith("bd"):
        if not plot_types["bunchdavies"]:
            continue
    elif file.endswith("off"):
        if not plot_types["off"]:
            continue
    else:
        if not plot_types["hartree"]:
            continue

    with open(file + ".quick", 'rb') as f:
        quickdata = pickle.load(f)

    for key in quickdata:
        if key in plot_data:
            plot_data[key].append(quickdata[key])
    if "efolds" not in quickdata:
        plot_data["efolds"].append(0.0)

# Convert plot data to numpy arrays
for key in plot_data:
    plot_data[key] = np.array(plot_data[key])

# Create PDF plots
# os.chdir(olddir)
# pdf_pages = PdfPages(args.outfilename)

fig = plot3d(plot_data["phi0"], plot_data["phi0dot"],
             plot_data["efolds"], '$N$')
plt.show()

# pdf_pages.savefig(fig)
# pdf_pages.close()
