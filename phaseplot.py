#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phaseplot.py

Plots the phase space trajectory in phi, phidot for multiple runs

Usage:
python phaseplot.py filenames...

"""
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

outfile = "phase.pdf"

####################################
# Deal with command line arguments #
####################################
files = sys.argv[1:]

if len(files) < 1:
    print("No files provided!")
    sys.exit(1)


#################
# Load the data #
#################
datastore = []

for file in files:
    with open(file) as f:
        data = f.readlines()
    results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
    a, phi0, phi0dot = results[1:4]
    datastore.append((a, phi0, phi0dot))


############
# Plotting #
############

print(f"Creating {outfile}")

# Set up the PDF
pdf_pages = PdfPages(outfile)
plt.rcParams["font.family"] = "serif"

# Create the canvas
canvas = plt.figure(figsize=(14.0, 14.0), dpi=100)
plt.subplot(1, 1, 1)

# Draw every series
for a, phi0, phi0dot in datastore:
    plt.plot(phi0, phi0dot)

# Apply labels
plt.xlabel(r"$\phi_0$")
plt.ylabel(r"$\dot{\phi}_0$")

# Finish the file
pdf_pages.savefig(canvas)
pdf_pages.close()
print("Finished!")
