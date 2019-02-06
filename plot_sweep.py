#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from evolver.model import Model
from evolver.utilities import load_data, analyze

####################################
# Deal with command line arguments #
####################################
parser = argparse.ArgumentParser(description="Plot data from a sweep")
parser.add_argument("filename", help="Base of the output name to read data in from")
parser.add_argument("outfilename", help="Filename to output to (include .pdf)")
args = parser.parse_args()


def phase_plotter(phis, phidots, cs, infls, cname):
    fig = plt.figure(figsize=(7.0, 7.0), dpi=100)
    sc = plt.scatter(phis, phidots, s=30.0+infls*40.0, c=cs,
                     cmap=cm.cool, marker='o', linewidth=0.0)
    plt.xlabel(r'$\phi$')
    plt.ylim((0.01, 0.03))
    plt.ylabel(r'$\dot{\phi}$')
    cbr = plt.colorbar(sc)
    cbr.set_label(cname, rotation=270, fontsize=8)
    return fig

# Specify the critical number of efolds above/below which we determine
# sufficient/insufficient inflation
Nef_crit = 65.0

# Suck up the data
with open(args.filename + "-info.txt") as f:
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
    model = Model.load(file + ".params")
    params = model.eomparams
    results = load_data(file)
    details = analyze(results["a"], results["epsilon"])

    # Store all of the data we wish to plot
    plot_data["phi0"].append(results["phi0"][0])
    plot_data["phi0dot"].append(results["phi0dot"][0])
    plot_data["H"].append(results["H"][0])
    plot_data["rho"].append(results["rho"][0])
    plot_data["deltarho2"].append(results["deltarho2"][0])
    plot_data["phi2pt"].append(results["phi2pt"][0])
    plot_data["kappa"].append(params.kappa)
    if "efolds" in details:
        plot_data["efolds"].append(details['efolds'])
    else:
        plot_data["efolds"].append(0.0)
    # Did sufficient inflation occur?
    if details['efolds'] >= Nef_crit:
        plot_data["infl"].append(1)
    else:
        plot_data["infl"].append(0)

# Convert plot data to numpy arrays
for key in plot_data:
    plot_data[key] = np.array(plot_data[key])

# Create PDF plots
pdf_pages = PdfPages(args.outfilename)
#
fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
                    plot_data["efolds"],
                    plot_data["infl"], '$N_{ef}$')
pdf_pages.savefig(fig)
#
# fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
#                     plot_data["rho"] + plot_data["deltarho2"],
#                     plot_data["infl"], r'$\rho + \delta \rho^{2}$')
# pdf_pages.savefig(fig)
# #
fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
                    plot_data["deltarho2"] / plot_data["rho"],
                    plot_data["infl"], r'$\delta \rho^{2} / \rho$')
pdf_pages.savefig(fig)
#
# fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
#                     plot_data["phi2pt"]/(plot_data["kappa"]/(2*np.pi))**2,
#                     plot_data["infl"],
#                     r'$<(\delta\phi)^{2}>_{t_{0}}/\frac{\kappa^{2}}{4\pi^{2}}$')
# pdf_pages.savefig(fig)
#
pdf_pages.close()
