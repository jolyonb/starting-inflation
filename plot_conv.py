#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
import os
import statistics
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as interpolate
from matplotlib.backends.backend_pdf import PdfPages
from evolver.model import Model

# Parse the filename
filename = "data/converge"
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
        fn, phi0, phi0dot, Rmax = line.strip().split("\t")
        data.append((fn, Rmax))

# Read the data files for each run in the sweep
plot_data = {
  "filename": [],
  "Rmaxfactor": [],
  "lna": [],
  "phi2pt": [],
  "phi2ptdt": [],
  "phi2ptgrad": [],
  "psi2pt": []
}
for file, Rmax in data:
    # Parameters
    model = Model.load(file + ".params")
    params = model.eomparams

    # File 1
    with open(file + ".dat") as f:
        data = f.readlines()
    results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
    a = results[1]

    # File 2
    with open(file + ".dat2") as f:
        data2 = f.readlines()
    results2 = np.array([list(map(float, line.split(", "))) for line in data2]).transpose()
    (_, _, _, _, phi2pt, phi2ptdt, phi2ptgrad, psi2pt, _,
        _, _, _, _, _, _, _) = results2[1:]

    # Save the data
    plot_data["filename"].append(file)
    plot_data["Rmaxfactor"].append(model.parameters["Rmaxfactor"])
    plot_data["lna"].append(np.log(a))
    plot_data["phi2pt"].append(phi2pt)
    plot_data["phi2ptdt"].append(phi2ptdt)
    plot_data["phi2ptgrad"].append(phi2ptgrad)
    plot_data["psi2pt"].append(psi2pt)

def create_cover_sheet(canvas, stats, Rmaxfactors):
    # Create a plot on the canvas
    ax = canvas.add_subplot(1, 1, 1)

    # Add the text we want
    ax.text(0.05, 0.95, r'$\phi_0$ = ' + str(phi0))
    ax.text(0.05, 0.90, r'$\dot{\phi}_0$ = ' + str(phi0dot))
    Rmax = [str(round(i, 2)) for i in Rmaxfactors]
    ax.text(0.05, 0.85, r'$R_{max}$ values: ' + ', '.join(Rmax))
    ax.text(0.05, 0.80, f'$N_{{efolds}}$ = {stats["efolds"][0]:.2f} $\\pm$ {stats["efolds"][1]:.3f}')
    ax.text(0.05, 0.75, r'Two point functions reported at ' + str(reporting) + ' efolds')
    ax.text(0.05, 0.70, r'$\langle (\delta \phi)^2 \rangle$' + f' = {stats["phi2pt"][0]:.2e} $\\pm$ {stats["phi2pt"][1]:.2e}')
    ax.text(0.05, 0.65, r'$\langle (\delta \dot{\phi})^2 \rangle$' + f' = {stats["phi2ptdt"][0]:.2e} $\\pm$ {stats["phi2ptdt"][1]:.2e}')
    ax.text(0.05, 0.60, r'$\langle h^{ij} \partial_i \delta \phi \partial_j \delta \phi \rangle$' + f' = {stats["phi2ptgrad"][0]:.2e} $\\pm$ {stats["phi2ptgrad"][1]:.2e}')
    ax.text(0.05, 0.55, r'$\langle \psi^2 \rangle$' + f' = {stats["psi2pt"][0]:.2e} $\\pm$ {stats["psi2pt"][1]:.2e}')

    # Hide the ticks (this is an empty plot!)
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False)

def make_pdf(pages, stats, Rmaxfactors, filename):
    # Create the PDF file
    print(f"Creating {filename}")
    pdf_pages = PdfPages(filename)

    # Write the cover sheet
    plt.rcParams["font.family"] = "serif"
    canvas = plt.figure(figsize=(8.0, 8.0), dpi=70)
    create_cover_sheet(canvas, stats, Rmaxfactors)
    pdf_pages.savefig(canvas)

    # Create the plots
    for idx, page in enumerate(pages):
        # Create the canvas
        canvas = plt.figure(figsize=(14.0, 14.0), dpi=100)

        # Sort out the page configuration
        numfigs = len(page)
        if numfigs in [1, 2]:
            rows = 2
            cols = 1
        elif numfigs in [3, 4]:
            rows = 2
            cols = 2
        else:
            print(f"Page {idx+1} has too many figures; only 4 will be produced")
            rows = 2
            cols = 2
            page = page[0:4]

        # Create the figures
        for fig, definition in enumerate(page):
            # Define the plotting location
            plt.subplot(rows, cols, fig + 1)

            # Specify when to use scientific notation
            plt.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))

            # Create the plot
            maxx = 0
            for x_series, y_series, name in zip(definition['x'], definition['y'], definition['legends']):
                x_data = np.real(x_series)
                y_data = np.real(y_series)
                plt.plot(x_data, y_data, label=f'{name:.2f}')
                if x_data[-1] > maxx:
                    maxx = x_data[-1]

            # Set the plot range
            if definition['x_range']:
                plt.xlim(*definition['x_range'])
            else:
                plt.xlim((0, maxx))
            if definition['y_range']:
                plt.ylim(*definition['y_range'])

            # Apply labels
            plt.xlabel(definition['x_label'])
            plt.ylabel(definition['y_label'])

            # Add a legend
            plt.legend()

        # Save the page
        pdf_pages.savefig(canvas)

    # Finish the file
    pdf_pages.close()
    print("Finished!")

def define_fig(x_data, y_data,
               x_label=r"$\ln(a)$", y_label=None,
               x_range=None, y_range=None,
               legends=None):
    """Constructs data for a figure"""
    return {
        'x': x_data,
        'y': y_data,
        'x_label': x_label,
        'y_label': y_label,
        'x_range': x_range,
        'y_range': y_range,
        'legends': legends
    }

def early(data, range=(0, 8)):
    """Restricts the plotting range in x to the given range"""
    return {**data, 'x_range': range}

# Construct the data for the plots
phi2ptplot = define_fig(x_data=plot_data["lna"],
                        y_data=plot_data["phi2pt"],
                        y_label=r'$\langle (\delta \phi)^2 \rangle$',
                        legends=plot_data["Rmaxfactor"])

phi2ptdtplot = define_fig(x_data=plot_data["lna"],
                          y_data=plot_data["phi2ptdt"],
                          y_label=r'$\langle (\delta \dot{\phi})^2 \rangle$',
                          legends=plot_data["Rmaxfactor"])

phi2ptgradplot = define_fig(x_data=plot_data["lna"],
                            y_data=plot_data["phi2ptgrad"],
                            y_label=r'$\langle h^{ij} \partial_i \delta \phi \partial_j \delta \phi \rangle$',
                            legends=plot_data["Rmaxfactor"])

psi2ptplot = define_fig(x_data=plot_data["lna"],
                        y_data=plot_data["psi2pt"],
                        y_label=r'$\langle \psi^2 \rangle$',
                        legends=plot_data["Rmaxfactor"])

# Compute statistics for cover page
reporting = 8

# Efolds
efolds = []
for avals in plot_data["lna"]:
    efolds.append(avals[-1])

# Two point phi
twoptphis = []
for i in range(len(plot_data["lna"])):
    x = plot_data["lna"][i]
    y = plot_data["phi2pt"][i]
    interp = interpolate(x, y)
    twoptphis.append(float(interp(reporting)))

# Two point phidot
twoptphidots = []
for i in range(len(plot_data["lna"])):
    x = plot_data["lna"][i]
    y = plot_data["phi2ptdt"][i]
    interp = interpolate(x, y)
    twoptphidots.append(float(interp(reporting)))

# Two point phigrad
twoptphigrads = []
for i in range(len(plot_data["lna"])):
    x = plot_data["lna"][i]
    y = plot_data["phi2ptgrad"][i]
    interp = interpolate(x, y)
    twoptphigrads.append(float(interp(reporting)))

# Two point psi
twoptpsis = []
for i in range(len(plot_data["lna"])):
    x = plot_data["lna"][i]
    y = plot_data["psi2pt"][i]
    interp = interpolate(x, y)
    twoptpsis.append(float(interp(reporting)))

def makestats(vals):
    return statistics.mean(vals), statistics.stdev(vals), max(vals) - min(vals)

stats = {
    "efolds": makestats(efolds),
    "phi2pt": makestats(twoptphis),
    "phi2ptdt": makestats(twoptphidots),
    "phi2ptgrad": makestats(twoptphigrads),
    "psi2pt": makestats(twoptpsis),
}

# Define the PDF layout
pages = [
    [early(phi2ptplot), phi2ptplot],
    [early(phi2ptdtplot), phi2ptdtplot],
    [early(phi2ptgradplot), phi2ptgradplot],
    [early(psi2ptplot), psi2ptplot]
]

# Construct the PDF
os.chdir(olddir)
make_pdf(pages, stats, plot_data["Rmaxfactor"], "convergence.pdf")
