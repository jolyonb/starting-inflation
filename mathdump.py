#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creates a data file for mathematica containing results from a sweep
"""
import pickle
import os
import argparse
import numpy as np

####################################
# Deal with command line arguments #
####################################
parser = argparse.ArgumentParser(description="Convert sweep data to a Mathematica file")
parser.add_argument("filename", help="Base of the output name to read data in from")
args = parser.parse_args()

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
# Store the results in this list:
fulldata = []

for file, phi0, phi0dot in data:
    with open(file + ".quick", 'rb') as f:
        quickdata = pickle.load(f)

    # The data fields we'll pick out from quickdata
    plot_data = {
      "phi0": 0.0,
      "phi0dot": 0.0,
      "H": 0.0,
      "rho": 0.0,
      "deltarho2": 0.0,
      "phi2pt": 0.0,
      "psirms": 0.0,
      "efolds": 0.0,
      "kappa": 0.0,
      "infl": 0
    }

    for key in quickdata:
        if key in plot_data:
            plot_data[key] = quickdata[key]

    # Add in any extra details about this run
    plot_data["filename"] = file
    # Type:
    # 0 = Hartree Off
    # 1 = Hartree On, Bunch-Davies
    # 2 = Hartree On, Perturbed
    if "-off" in file:
        plot_data["type"] = 0
    elif "-bd" in file:
        plot_data["type"] = 1
    else:
        plot_data["type"] = 2
    plot_data['infl'] = 1 if plot_data['infl'] else 0

    fulldata.append(plot_data)

# All data from the sweep is now stored in fulldata
# Output it to a file!
os.chdir(olddir)
template = "{phi0},{phi0dot},{H},{rho},{deltarho2},{phi2pt},{psirms},{efolds},{kappa},{infl},{type},{filename}\n"
with open(filename + "-math.csv", "w") as f:
    for entry in fulldata:
        f.write(template.format(**entry))

print("Data written to " + filename + "-math.csv")
