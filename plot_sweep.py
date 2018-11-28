#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from evolver.analysis import load_data, analyze

def phase_plotter(phis, phidots, cs, infls, cname):
    fig = plt.figure(figsize=(7.0, 7.0), dpi=100)
    sc = plt.scatter(phis, phidots, s=10.0+infls*50.0, c=cs,
                     cmap=cm.YlOrRd, marker='o', linewidth=0.0)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\dot{\phi}$')
    cbr = plt.colorbar(sc)
    cbr.set_label(cname, rotation=270, fontsize=8)
    return fig

# Output file for the sweep plots:
sweep_plt = "sweep_hOff.pdf"

# The info file to read from
filename = "data/output-info.txt"

# Specify the critical number of efolds above/below which we determine
# sufficient/insufficient inflation
Nef_crit = 60.0

# Suck up the data
with open(filename) as f:
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
  "infl": []
}
for file, phi0, phi0dot in data:
    results = load_data(file)
    details = analyze(results["a"], results["adot"], results["addot"])

    # Store all of the data we wish to plot
    plot_data["phi0"].append(results["phi0"][0])
    plot_data["phi0dot"].append(results["phi0dot"][0])
    plot_data["H"].append(results["H"][0])
    plot_data["rho"].append(results["rho"][0])
    plot_data["deltarho2"].append(results["deltarho2"][0])
    plot_data["phi2pt"].append(results["phi2pt"][0])
    if "efolds" in details:
        plot_data["efolds"].append(results["phi0"][0])
    else:
        plot_data["efolds"].append(0.0)
    # Did sufficient inflation occur?
    if plot_data["efolds"] >= Nef_crit:
        plot_data["infl"].append(1)
    else:
        plot_data["infl"].append(0)

# Convert plot data to numpy arrays
for key in plot_data:
    plot_data[key] = np.array(plot_data[key])

# Create PDF plots
pdf_pages = PdfPages(sweep_plt)
#
fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
                    plot_data["efolds"],
                    plot_data["infl"], '$N_{ef}$')
pdf_pages.savefig(fig)
#
fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
                    plot_data["rho"] + plot_data["deltarho2"],
                    plot_data["infl"], r'$\rho + \delta \rho^{2}$')
pdf_pages.savefig(fig)
#
fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
                    plot_data["deltarho2"] / plot_data["rho"],
                    plot_data["infl"], r'$\delta \rho^{2} / \rho$')
pdf_pages.savefig(fig)
#
fig = phase_plotter(plot_data["phi0"], plot_data["phi0dot"],
                    plot_data["phi2pt"]/(plot_data["H"]/(2*np.pi))**2,
                    plot_data["infl"],
                    r'$<(\delta\phi)^{2}>_{t_{0}}/\frac{H_{0}^{2}}{4\pi^{2}}$')
pdf_pages.savefig(fig)
#
pdf_pages.close()
