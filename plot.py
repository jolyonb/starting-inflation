#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
from run import params
from evolver.initialize import unpack
from math import pi
from matplotlib.backends.backend_pdf import PdfPages
from evolver.eoms import N_efolds
from enum import Enum

####################################
# Deal with command line arguments #
####################################
parser = argparse.ArgumentParser(description="Plot data from a run")
parser.add_argument("filename", help="Base of the filename to read data in from")
parser.add_argument("outfilename", help="Filename to output to (include .pdf)")
args = parser.parse_args()


#################
# Load the data #
#################
# File 1
with open(args.filename + ".dat") as f:
    data = f.readlines()
results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
t = results[0]
(a, adot, phi0, phi0dot, phiA, phidotA,
 psiA, phiB, phidotB, psiB) = unpack(results[1:], params.total_wavenumbers)

# File 2
with open(args.filename + ".dat2") as f:
    data2 = f.readlines()
results2 = np.array([list(map(float, line.split(", "))) for line in data2]).transpose()
(H, Hdot, addot, phi0ddot, phi2pt, phi2ptdt, phi2ptgrad, psi2pt, rho,
    deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd) = results2[1:]

# Construct the perturbative modes
# \ell = 0
phi_l0 = [None] * params.k_modes
phidot_l0 = [None] * params.k_modes
psi_l0 = [None] * params.k_modes
for i in range(params.k_modes):
    phi_l0[i] = params.poscoeffs[0][i] * phiA[i] + params.velcoeffs[0][i] * phiB[i]
    phidot_l0[i] = params.poscoeffs[0][i] * phidotA[i] + params.velcoeffs[0][i] * phidotB[i]
    psi_l0[i] = params.poscoeffs[0][i] * psiA[i] + params.velcoeffs[0][i] * psiB[i]

# Just one of the m_\ell modes for \ell = 1
phi_l1 = [None] * (params.k_modes-1)
phidot_l1 = [None] * (params.k_modes-1)
psi_l1 = [None] * (params.k_modes-1)
for i in range(params.k_modes-1):
    phi_l1[i] = params.poscoeffs[1][0][i] * phiA[i] + params.velcoeffs[1][0][i] * phiB[i]
    phidot_l1[i] = params.poscoeffs[1][0][i] * phidotA[i] + params.velcoeffs[1][0][i] * phidotB[i]
    psi_l1[i] = params.poscoeffs[1][0][i] * psiA[i] + params.velcoeffs[1][0][i] * psiB[i]

# Make a big list of all the perturbative modes
# deltaphi = phi_l0 + phi_l1
# deltaphidot = phidot_l0 + phidot_l1
# psi = psi_l0 + psi_l1
# num_modes = params.total_wavenumbers
# k_modes = params.all_wavenumbers

# Just use the \ell = 0 modes
deltaphi = phi_l0
deltaphidot = phidot_l0
psi = psi_l0
num_modes = params.k_modes
k_modes = params.k_grids[0]


######################
# Plotting Functions #
######################

class PlotStyle(Enum):
    LINEAR = 1
    LOG10 = 2

def create_cover_sheet(canvas):
    # Create a plot on the canvas
    ax = canvas.add_subplot(1, 1, 1)

    # Add the text we want
    ax.text(0.05, 0.95, r'$K$ = 0.0')
    ax.text(0.05, 0.90, (r'$\frac{\delta\rho^{(2)}(0)}{\rho(0)}$ = '
                         + str(round(deltarho2[0]/rho[0], 3))))
    ax.text(0.05, 0.85, r'$R_{\rm max} H(0)$ = ' + str(round(params.Rmax, 1)))
    ax.text(0.05, 0.80, r'$\lambda$ = ' + str(params.model.lamda))
    ax.text(0.05, 0.75, r'$\frac{\kappa}{H(0)}$ = ' + str(round(params.kappa/H[0], 1)))
    ax.text(0.05, 0.70, r'$\phi_0$ = ' + str(phi0[0]))
    ax.text(0.05, 0.65, r'$\dot{\phi}_0$ = ' + str(phi0dot[0]))
    ax.text(0.05, 0.60, r'$a(0)$ = ' + str(a[0]))
    ax.text(0.05, 0.55, r'$H(0)$ = ' + str(round(H[0], 6)))
    ax.text(0.05, 0.50, r'$\frac{\kappa^2}{4\pi^2}$ = ' + str(round((params.kappa**2/4/pi**2), 6)))
    ax.text(0.05, 0.45, r'$\langle \delta\phi^2 \rangle$ = ' + str(round(phi2pt[0], 6)))
    ax.text(0.05, 0.40, (r'$\frac{\langle \delta\phi^2 \rangle}{(\kappa^2/4\pi^2)}$ = '
                         + str(round(phi2pt[0] / (params.kappa**2/4/pi**2), 6))))
    ax.text(0.05, 0.35, r'$\langle \psi^2 \rangle$ = ' + str(round(psi2pt[0], 6)))
    ax.text(0.05, 0.30, r'$N_{\rm e-folds}$ = ' + str(round(N_efolds(a[-1]), 2)))
    ax.text(0.05, 0.25, r'$n_{\rm max}$ = ' + str(round(params.k_modes, 1)))

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

def make_pdf(pages, filename):
    # Create the PDF file
    print(f"Creating {filename}")
    pdf_pages = PdfPages(filename)

    # Write the cover sheet
    plt.rcParams["font.family"] = "serif"
    canvas = plt.figure(figsize=(8.0, 8.0), dpi=70)
    create_cover_sheet(canvas)
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

            # Determine the plotting function
            if definition['y_type'] == PlotStyle.LOG10:
                plotter = plt.semilogy
            elif definition['y_type'] == PlotStyle.LINEAR:
                plotter = plt.plot
            else:
                print(f"Bad plotting instruction on page {idx+1}, figure {fig+1}")

            # Do we have one data series, or a list?
            data = definition['y']
            if not isinstance(data, list):
                data = [data]

            # Create the plot
            for y_series in data:
                y_data = np.real(y_series)
                plotter(definition['x'], y_data)

            # Set the plot range
            if definition['x_range']:
                plt.xlim(*definition['x_range'])
            else:
                plt.xlim((definition['x'][0], definition['x'][-1]))
            if definition['y_range']:
                plt.ylim(*definition['y_range'])

            # Apply labels
            plt.xlabel(definition['x_label'])
            plt.ylabel(definition['y_label'])

        # Save the page
        pdf_pages.savefig(canvas)

    # Finish the file
    pdf_pages.close()
    print("Finished!")

def define_fig(x_data, y_data,
               x_label=r"$\ln(a)$", y_label=None,
               x_range=None, y_range=None,
               y_type=PlotStyle.LINEAR):
    """Constructs data for a figure"""
    return {
        'x': x_data,
        'y': y_data,
        'x_label': x_label,
        'y_label': y_label,
        'x_range': x_range,
        'y_range': y_range,
        'y_type': y_type
    }

def early(data, range=(0, 7)):
    """Restricts the plotting range in x to the given range"""
    return {**data, 'x_range': range}


####################
# Plot Definitions #
####################

# Note that it is relatively quick to make all these definitions
# The slow part is the plotting of whatever is actually included in the PDF
# So, it is convenient to define everything we could ever want to plot here!

# x-axis for all the plots
lna = np.log(a)

# Background quantities
Hplot = define_fig(x_data=lna, y_data=H, y_label='H')
Hdotplot = define_fig(x_data=lna, y_data=Hdot, y_label=r'$\dot{H}$')
phi0plot = define_fig(x_data=lna, y_data=phi0, y_label=r'$\phi_0$')
epsilonplot = define_fig(x_data=lna, y_data=-Hdot/H**2, y_label=r'$\epsilon$')

# Energies
rhoplot = define_fig(x_data=lna,
                     y_data=rho,
                     y_label=r'$\rho$',
                     y_type=PlotStyle.LOG10)
deltarho2plot = define_fig(x_data=lna,
                           y_data=deltarho2,
                           y_label=r'$\delta\rho^{(2)}$',
                           y_type=PlotStyle.LOG10)
energyratio = define_fig(x_data=lna,
                         y_data=deltarho2/rho,
                         y_label=r'$\frac{\delta\rho^{(2)}}{\rho}$',
                         y_type=PlotStyle.LOG10)

# \delta\phi_k power spectrum
data = []
for i in range(num_modes):
    data.append(1/(2*pi**2) * k_modes[i]**3 * deltaphi[i] * np.conj(deltaphi[i]))
deltaphiplots = define_fig(x_data=lna,
                           y_data=data,
                           y_label=r'$\frac{k^3}{2\pi^2} |\delta \phi_{k}|^2$',
                           y_type=PlotStyle.LOG10)

# \psi power spectrum
data = []
for i in range(num_modes):
    data.append(1/(2*pi**2) * k_modes[i]**3 * psi[i] * np.conj(psi[i]))
psiplots = define_fig(x_data=lna,
                      y_data=data,
                      y_label=r'$\frac{k^3}{2\pi^2} |\psi_{k}|^2$',
                      y_type=PlotStyle.LOG10)

# \psi real and imaginary parts
data = [np.real(line) for line in psi]
psireplots = define_fig(x_data=lna,
                        y_data=data,
                        y_label=r'$\mathrm{Re}(\psi_{k})$',
                        y_range=[-0.1, 0.1])
data = [np.imag(line) for line in psi]
psiimplots = define_fig(x_data=lna,
                        y_data=data,
                        y_label=r'$\mathrm{Im}(\psi_{k})$',
                        y_range=[-0.1, 0.1])

# \psi RMS value
psirmsplot = define_fig(x_data=lna,
                        y_data=np.sqrt(psi2pt),
                        y_label=r'$\sqrt{\langle \psi^2 \rangle}$')

# \psi constraint violation
real_data = []
imag_data = []
for i in range(num_modes):
    constraint = 1/2*(phi0ddot*deltaphi[i] - phi0dot*deltaphidot[i]) / (Hdot + k_modes[i]**2/(a*a))
    violation = constraint - psi[i]
    real_data.append(np.real(violation))
    imag_data.append(np.imag(violation))
psi_violations_real = define_fig(x_data=lna,
                                 y_data=real_data,
                                 y_label=r'$\mathrm{Re}(C_k)$',
                                 y_range=(-1, 1))
psi_violations_imag = define_fig(x_data=lna,
                                 y_data=imag_data,
                                 y_label=r'$\mathrm{Im}(C_k)$',
                                 y_range=(-1, 1))

# Other plots we may want to define:
# Hubble violation
# deltarho2_kinetic
# deltarho2_kinetic / \dot{\phi}_0^2
# R


###########################
# PDF Layout and Creation #
###########################

# Lay out the figures in pages
# We recommend commenting out pages that you don't want, rather than deleting them
pages = [
    [Hplot, Hdotplot, phi0plot, epsilonplot],
    [early(rhoplot), early(deltarho2plot), early(energyratio)],
    [rhoplot, deltarho2plot, energyratio],
    [early(deltaphiplots), deltaphiplots],
    [early(psiplots), psiplots],
    [early(psireplots), early(psiimplots)],
    [early(psirmsplot), psirmsplot],
    [psi_violations_real, psi_violations_imag],
    [early(psi_violations_real), early(psi_violations_imag)]
]

# Construct the PDF
make_pdf(pages, args.outfilename)
