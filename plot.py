#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
import evolver.eoms
from run import params
from evolver.initialize import unpack, make_initial_data, Parameters
from math import exp, pi
from matplotlib.backends.backend_pdf import PdfPages
from evolver.analysis import analyze
from evolver.eoms import N_efolds

plt.rcParams["font.family"] = "serif"
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
with open(params.filename+".dat") as f:
    data = f.readlines()
with open(params.filename+".dat2") as f:
    data2 = f.readlines()

# Process the data
results = np.array([list(map(float, line.split(", "))) for line in data]).transpose()
results2 = np.array([list(map(float, line.split(", "))) for line in data2]).transpose()

# Unpack the data
t = results[0]
a, adot, phi0, phi0dot, phiA, phidotA, psiA, phiB, phidotB, psiB = unpack(results[1:], params.total_wavenumbers)

(H, Hdot, addot, phi0ddot, hpotential0, hgradient0, hkinetic0, psi2pt, rho, 
    deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd) = results2[1:]

phi = [None] * params.k_modes
phidot = [None] * params.k_modes
psi = [None] * params.k_modes
phi_l1 = [None] * (params.k_modes-1)
phidot_l1 = [None] * (params.k_modes-1)
psi_l1 = [None] * (params.k_modes-1)

# Attach coefficients
for i in range(params.k_modes):
    params.poscoeffs[0][i]
    phi[i] = params.poscoeffs[0][i] * phiA[i] + params.velcoeffs[0][i] * phiB[i]
    phidot[i] = params.poscoeffs[0][i] * phidotA[i] + params.velcoeffs[0][i] * phidotB[i]
    psi[i] = params.poscoeffs[0][i] * psiA[i] + params.velcoeffs[0][i] * psiB[i]

for i in range(params.k_modes-1):
    params.poscoeffs[0][i]
    phi_l1[i] = params.poscoeffs[1][0][i] * phiA[i] + params.velcoeffs[1][0][i] * phiB[i]
    phidot_l1[i] = params.poscoeffs[1][0][i] * phidotA[i] + params.velcoeffs[1][0][i] * phidotB[i]
    psi_l1[i] = params.poscoeffs[1][0][i] * psiA[i] + params.velcoeffs[1][0][i] * psiB[i]

# function to calculate script_M Eq. 1.18 from Dave's "Coupled Equations of Motion"
def script_M(i):
    return (Vd + 2*H*phi0dot) / (Hdot + params.k_grids[0][i]*params.k_grids[0][i]/(a*a))

def basic_plot(xvals, yvals, plottype, num_plots, position):
    '''
    Constructs a basic plot of your quanitity of interest

    Arguments:
        * xvals: variable plotted as x
        * yvals: variable plotted as y
        * plottype: choose how to plot your two variables
        * position: position of the plot on page (Python plots top to bottom, left to right)
        * total_plots: total number of plots to appear on this page

    Returns your plot
    '''
    if plottype == 0:
        yvals = yvals
    elif plottype == 1:
        xvals = np.log(xvals)
    elif plottype == 2:
        yvals = np.log(yvals)
    elif plottype == 3:
        xvals = np.log(xvals)
        yvals = np.log(yvals)
    else:
        print("your request is not a valid plot format, please try again")

    if num_plots == 1:
        columns = 1
    elif num_plots == 2:
        columns = 1
    elif num_plots > 2 and num_plots <= 4:
        columns = 2
    else:
        print ("too many plots per page, downsize!")

    plt.subplot(2,columns,position)
    plt.plot(xvals, yvals)
    # xlabel = str(xvals)
    # plt.xlabel(xlabel)

    return 

def cover_sheet():
    ax = fig0.add_subplot(111)
    ax.text(0.05,0.95, '$K$=0.0')
    ax.text(0.05,0.90, '$\\delta \\rho^{(2)}_{0}/\\rho_{0}$='+str(round(deltarho2[0]/rho[0],3)))
    ax.text(0.05,0.85, '$R_{max} (H_{0}^{-1})$='+str(round(params.Rmax,1)))
    ax.text(0.05,0.80, '$\\lambda$=' + str(params.model.lamda))
    ax.text(0.05,0.75, '$\\kappa  (H_{0})$='+str(round(params.kappa/H[0],1)))
    ax.text(0.05,0.70, '$\\phi_{0}$='+str(phi0[0]))
    ax.text(0.05,0.65, '$\\dot{\\phi}_{0}$='+str(phi0dot[0]))
    ax.text(0.05,0.60, '$a_{0}$='+str(a[0]))
    ax.text(0.05,0.55, '$H_{0}$='+str(round(H[0],6)))
    ax.text(0.05,0.50, '$\\frac{\kappa^{2}}{4\\pi^{2}}$='+str(round((params.kappa**2/4/pi**2),6)))
    ax.text(0.05,0.45, '$\langle (\delta\phi)^2 \\rangle$='+str(round(hpotential0[0],6)))
    ax.text(0.05,0.40, '$\langle (\delta\psi)^2 \\rangle$='+str(round(psi2pt[0],6)))
    ax.text(0.05,0.35, '$\\frac{\langle (\delta\phi)^2 \\rangle}{\\frac{\kappa^{2}}{4\\pi^{2}}}=$'+str(round(hpotential0[0] / (params.kappa**2/4/pi**2),6)))
    ax.text(0.05, 0.30, '$N_{e-folds}$='+str(round(N_efolds(a[-1]),2)))
    ax.text(0.05, 0.25, '$n_{max}$='+str(round(params.k_modes,1)))
    ax.set_xticklabels([])
    ax.set_yticklabels([])


##########################################
# create pdf output of plots 
##########################################

strplots = "BD"+'_plots'+'.pdf'
pdf_pages = PdfPages(strplots)



# print initial conditions
fig0 = plt.figure(figsize=(8.0,8.0),dpi=70)
cover_sheet()
pdf_pages.savefig(fig0)    



# plot background quantities over all of inflation
fig1 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig1_numplots = 4
basic_plot(a, H, 1, fig1_numplots, 1)
basic_plot(a, Hdot, 1, fig1_numplots, 2)
basic_plot(a, phi0, 1, fig1_numplots, 3)
basic_plot(a, -Hdot/(H*H), 1, fig1_numplots, 4)
pdf_pages.savefig(fig1)   



# plot energy densities over early times
# TO DO: include xlim here
# fig2 = plt.figure(figsize=(14.0,14.0),dpi=100)
# basic_plot(a, rho, 3, 1)
# basic_plot(a, deltarho2, 3, 2)
# basic_plot(a, deltarho2/rho, 3, 3)
# pdf_pages.savefig(fig2)   



# plot energy densities over all of inflation
fig3 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig3_numplots = 3
basic_plot(a, rho, 3, fig3_numplots, 1)
basic_plot(a, deltarho2, 3, fig3_numplots, 2)
basic_plot(a, deltarho2/rho, 3, fig3_numplots, 3)
pdf_pages.savefig(fig3) 



# plot dimensionless power spectrum for phi modes
fig4 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig4_numplots = 1
for i in range(params.k_modes):
    basic_plot(a,((1/(2*pi*pi)) * (params.k_grids[0][i] * params.k_grids[0][i] * params.k_grids[0][i]) * np.real(phi[i]*np.conj(phi[i]))), 3, fig4_numplots, 1)
# TO DO: Ask Jolyon if possible to rewrite this to avoid warning message, despite the fact that nothing untoward happens and plot is precisely what is wanted
# plt.ylabel('$ \\frac{k^{3}}{2\\pi^{2}} |\\phi_{k}|^{2}$')
pdf_pages.savefig(fig4) 



# plot dimensionless power spectrum for psi modes
fig5 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig5_numplots = 1
for i in range(params.k_modes):
    basic_plot(a,((1/(2*pi*pi)) * (params.k_grids[0][i] * params.k_grids[0][i] * params.k_grids[0][i]) * np.real(psi[i]*np.conj(psi[i]))), 3, fig5_numplots,1)
# plt.ylabel('$ \\frac{k^{3}}{2\\pi^{2}} |\\Psi_{k}|^{2}$')
pdf_pages.savefig(fig5) 



# plot psi RMS
fig6 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig6_numplots = 1
basic_plot(a, np.sqrt(psi2pt), 1, fig6_numplots, 1)
# plt.ylabel('$\sqrt{\langle (\Psi)^2 \\rangle}$')
pdf_pages.savefig(fig6) 



# plot phi constraint
fig7 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig7_numplots = 1
for i in range(params.k_modes):
    psi_constraint_complex = (0.5*(phi0ddot*phi[i]-phi0dot*phidot[i])) / (Hdot + params.k_grids[0][i]*params.k_grids[0][i]/(a*a)) - psi[i]
    basic_plot(a, np.real(psi_constraint_complex), 1, fig7_numplots, 1)
    basic_plot(a, np.imag(psi_constraint_complex), 1, fig7_numplots, 1)
# plt.ylabel('$ C_{k} $')
pdf_pages.savefig(fig7)



pdf_pages.close()