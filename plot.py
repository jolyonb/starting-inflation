#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import argparse
import evolver.eoms
from run import params
from evolver.initialize import unpack, make_initial_data, Parameters
from math import exp
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

(H, Hdot, addot, phi0ddot, hpotential0, hgradient0, hkinetic0, psi2pt, rho, 
    deltarho2, hubble_violation, V, Vd, Vdd, Vddd, Vdddd) = results2[1:]

# check that Mathematica notebook gives same output
# print ("twopointBD:", hpotential0[0])
# print ("deltarho2:", deltarho2[0])
# print ("deltarho2_potential:", (hpotential0[0]*Vdd[0]/2))
# print ("deltarho2_kinetic:", (hkinetic0[0]/2))

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


##########################################
# create pdf output of plots 
##########################################
strplots = "BD"+'_plots'+'.pdf'
pdf_pages = PdfPages(strplots)

# print initial conditions
fig0 = plt.figure()

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
ax.text(0.05,0.50, '$\\frac{H_{0}^{2}}{4\\pi^{2}}$='+str(round(H[0]*H[0]/(2*2*np.pi*np.pi),6)))
ax.text(0.05,0.45, '$ratio$='+str(round(hpotential0[0] / (H[0]*H[0]/(2*2*np.pi*np.pi)),2)))
ax.text(0.05, 0.40, '$N_{e-folds}$='+str(round(N_efolds(a[-1]),2)))
ax.text(0.05, 0.35, '$n_{max}$='+str(round(params.k_modes,1)))

# for i in range(params.k_modes):
# 	ax.text(0.05, 0.30 - 0.05*i, '$\\delta\\phi$' + '$k$' + str(i+1) +
# 	'=' + str(round(np.real(phi[i][0]),4)) + "$+$" + str(round(np.imag(phi[i][0]),4))+ "$j$")

# for i in range(params.k_modes):
# 	ax.text(0.37, 0.90, '$l=0 \\ modes$')
# 	ax.text(0.37, 0.85 - 0.05*i, '$k$'+ str(i+1) + '$=$' + str(round(params.k_grids[0][i],5)))

# for i in range(params.k_modes):
# 	ax.text(0.37, 0.30 - 0.05*i, '$\\psi$k'+ str(i+1) +'=' + str(round(np.real(psi[i][0]),4)) + "$+$" + str(round(np.imag(psi[i][0]),4))+"$j$")

# for i in range(params.k_modes-1):
# 	ax.text(0.70,0.90, '$l=1 \\ modes$')
# 	ax.text(0.70, 0.85 - 0.05*i, '$k$'+ str(i+1) + '$=$' + str(round(params.k_grids[1][i],5)))

ax.set_xticklabels([])
ax.set_yticklabels([])

pdf_pages.savefig(fig0)    

# plot background quantities
fig1 = plt.figure(figsize=(14.0,14.0),dpi=100)

fig1_xlim = t[-1]

plt.subplot(2,2,1)
plt.plot(t, H)
plt.xlabel(' t ')
plt.ylabel('$ H $')
plt.xlim((0.0,fig1_xlim))
plt.ylim((0.0,0.01))

plt.subplot(2,2,2)
plt.plot(t, phi0)
plt.xlabel(' t ')
plt.ylabel('$\\phi$')
plt.xlim((0.0, fig1_xlim))

plt.subplot(2,2,3)
plt.plot(t, -Hdot/(H*H))
plt.xlabel(' t ')
plt.ylabel('$\\epsilon$')
plt.xlim((0.0, fig1_xlim))
plt.ylim((0.0,3.0))

pdf_pages.savefig(fig1)


# plot energy densities and ratios
fig2 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig2_xlim = t[-1]/20

plt.subplot(2,3,1)
plt.plot(t, np.log10(rho))
plt.xlabel(' t ')
plt.ylabel('$ \\log_{10}\\rho $')
plt.xlim((0.0, fig2_xlim))

plt.subplot(2,3,2)
plt.plot(t, np.log10(deltarho2))
plt.xlabel(' t ')
plt.ylabel('$ \\log_{10}\\delta\\rho_{(2)} $')
plt.xlim((0.0, fig2_xlim))

plt.subplot(2,3,3)
plt.plot(t, np.log10(deltarho2/rho))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2)}}{\\rho} $')
plt.xlim((0.0, fig2_xlim))

plt.subplot(2,3,4)
plt.plot(t, np.log10((hkinetic0/2.0)/deltarho2))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2, kinetic)}}{\\delta\\rho_{(2)}} $')
plt.xlim((0.0, fig2_xlim))

plt.subplot(2,3,5)
plt.plot(t, np.log10((hpotential0*Vdd/2.0)/deltarho2))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2, potential)}}{\\delta\\rho_{(2)}} $')
plt.xlim((0.0, fig2_xlim))

plt.subplot(2,3,6)
plt.plot(t, np.log10((0.5* hgradient0/(a*a))/deltarho2))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2, gradient)}}{\\delta\\rho_{(2)}} $')
plt.xlim((0.0, fig2_xlim))

pdf_pages.savefig(fig2)


# plot energy densities and ratios
fig3 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig3_xlim = t[-1]

plt.subplot(2,3,1)
plt.plot(t, np.log10(rho))
plt.xlabel(' t ')
plt.ylabel('$ \\log_{10}\\rho $')
plt.xlim((0.0, fig3_xlim))

plt.subplot(2,3,2)
plt.plot(t, np.log10(deltarho2))
plt.xlabel(' t ')
plt.ylabel('$ \\log_{10}\\delta\\rho_{(2)} $')
plt.xlim((0.0, fig3_xlim))

plt.subplot(2,3,3)
plt.plot(t, np.log10(deltarho2/rho))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2)}}{\\rho} $')
plt.xlim((0.0, fig3_xlim))

plt.subplot(2,3,4)
plt.plot(t, np.log10((hkinetic0/2.0)/deltarho2))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2, kinetic)}}{\\delta\\rho_{(2)}} $')
plt.xlim((0.0, fig3_xlim))

plt.subplot(2,3,5)
plt.plot(t, np.log10((hpotential0*Vdd/2.0)/deltarho2))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2, potential)}}{\\delta\\rho_{(2)}} $')
plt.xlim((0.0, fig3_xlim))

plt.subplot(2,3,6)
plt.plot(t, np.log10((0.5* hgradient0/(a*a))/deltarho2))
plt.xlabel(' t ')
plt.ylabel('$\\log_{10} \\ \\frac {\\delta\\rho_{(2, gradient)}}{\\delta\\rho_{(2)}} $')
plt.xlim((0.0, fig3_xlim))

pdf_pages.savefig(fig3)


# plot the dimensionless power spectrum for phi modes
fig4 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig4_xlim = t[-1]

plt.subplot(2,1,1)
for i in range(params.k_modes):
    plt.plot(t, np.log10((1/(2*np.pi*np.pi)) * (params.k_grids[0][i] * params.k_grids[0][i] * params.k_grids[0][i]) * np.real(phi[i]*np.conj(phi[i]))))
plt.xlabel(' t ')
plt.ylabel('$ \\frac{k^{3}}{2\\pi^{2}} |\\phi_{k}|^{2}$')
plt.xlim((0.0,60.0*np.sqrt(1e-6/params.model.lamda)))
plt.xlim((0.0,fig4_xlim/20))

plt.subplot(2,1,2)
for i in range(params.k_modes):
    plt.plot(t, np.log10((1/(2*np.pi*np.pi)) * (params.k_grids[0][i] * params.k_grids[0][i] * params.k_grids[0][i]) * np.real(phi[i]*np.conj(phi[i]))))
plt.xlabel(' t ')
plt.ylabel('$ \\frac{k^{3}}{2\\pi^{2}} |\\phi_{k}|^{2}$')
plt.xlim((0.0,fig4_xlim))

pdf_pages.savefig(fig4)


# plot the dimensionless power spectrum for Psi modes
fig5 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig5_xlim = t[-1]

plt.subplot(2,1,1)
for i in range(params.k_modes):
    plt.plot(t, np.log10((1/(2*np.pi*np.pi)) * (params.k_grids[0][i] * params.k_grids[0][i] * params.k_grids[0][i]) * np.real(psi[i]*np.conj(psi[i]))))
plt.xlabel(' t ')
plt.ylabel('$ \\frac{k^{3}}{2\\pi^{2}} |\\Psi_{k}|^{2}$')
plt.xlim((0.0,60.0*np.sqrt(1e-6/params.model.lamda)))
plt.xlim((0.0,fig5_xlim/20))

plt.subplot(2,1,2)
for i in range(params.k_modes):
    plt.plot(t, np.log10((1/(2*np.pi*np.pi)) * (params.k_grids[0][i] * params.k_grids[0][i] * params.k_grids[0][i]) * np.real(psi[i]*np.conj(psi[i]))))
plt.xlabel(' t ')
plt.ylabel('$ \\frac{k^{3}}{2\\pi^{2}} |\\Psi_{k}|^{2}$')
plt.xlim((0.0, fig5_xlim))

pdf_pages.savefig(fig5)


# plot phi constraint
fig6 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig6_xlim = t[-1]

plt.subplot(2,1,1)
for i in range(params.k_modes):
    psi_constraint_complex = (0.5*(phi0ddot*phi[i]-phi0dot*phidot[i])) / (Hdot + params.k_grids[0][i]*params.k_grids[0][i]/(a*a)) - psi[i]
    plt.plot(t, np.real(psi_constraint_complex))
    plt.plot(t, np.imag(psi_constraint_complex),'--')
plt.xlabel(' t ')
plt.xlim((0, 60.0*np.sqrt(1e-6/params.model.lamda)))
plt.ylabel('$ C_{k} $')

plt.subplot(2,1,2)
for i in range(params.k_modes):
    psi_constraint_complex = (0.5*(phi0ddot*phi[i]-phi0dot*phidot[i])) / (Hdot + params.k_grids[0][i]*params.k_grids[0][i]/(a*a)) - psi[i]
    plt.plot(t, np.real(psi_constraint_complex))
    plt.plot(t, np.imag(psi_constraint_complex),'--')
plt.xlabel(' t ')
plt.xlim((0, fig6_xlim))
plt.ylabel('$ C_{k} $')

pdf_pages.savefig(fig6)


#RMS value of the metric perturbations in position space analogous to 2pt for phi
fig7 = plt.figure(figsize=(14.0,14.0),dpi=100)
fig7_xlim = t[-1]

plt.subplot(2,1,1)
plt.plot ()
plt.xlabel(' t ')
plt.ylabel('$\sqrt{\langle (\Psi)^2 \\rangle}$')
plt.subplot(2,1,2)

pdf_pages.savefig(fig7)


pdf_pages.close()