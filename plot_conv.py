#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
import os
import numpy as np
import matplotlib.pyplot as plt
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
        fn, _, _, Rmax = line.strip().split("\t")
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

def make_pdf(pages, filename):
    # Create the PDF file
    print(f"Creating {filename}")
    pdf_pages = PdfPages(filename)

    # Write the cover sheet
    plt.rcParams["font.family"] = "serif"
    # canvas = plt.figure(figsize=(8.0, 8.0), dpi=70)
    # create_cover_sheet(canvas)
    # pdf_pages.savefig(canvas)

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
                plt.plot(x_data, y_data, label=str(name))
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
                        y_label=r'$\langle (\delta \psi)^2 \rangle$',
                        legends=plot_data["Rmaxfactor"])

print("E-folds by run:")

for avals in plot_data["lna"]:
    print(avals[-1])

# Define the PDF layout
pages = [
    [phi2ptplot, phi2ptdtplot],
    [phi2ptgradplot, psi2ptplot]
]

# Construct the PDF
os.chdir(olddir)
make_pdf(pages, "convergence.pdf")
