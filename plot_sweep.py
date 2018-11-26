#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the results from a parameter sweep
"""
from evolver.analysis import load_data, analyze

import swp_cls

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from evolver.eoms import slow_roll_epsilon

def phase_plotter(phis, phidots, cs, infls, cname):
    sc = plt.scatter(phis, phidots, s=10.0+infls*50.0, c=cs, cmap=cm.YlOrRd, marker='o', linewidth=0.0)
    plt.xlabel('$\phi $')
    plt.ylabel('$\dot{ \phi } $')
    cbr=plt.colorbar(sc)
    cbr.set_label(cname,rotation=270,fontsize=8)




#output file for the sweep plots:
filename_swp = "output_swp.dat"
sweep_plt = "sweep_hOff.pdf"

#specify the critical number of efolds above/below which we determine sufficient/insufficient inflation
Nef_crit = 60.0

#initialize the class which will hold all of the parameters and solutions of each run (each value in runs will be updated as we loop through phase space)
runs = swp_cls.runs([], [], [], [], [], [], [], [], [])

# Give it the info file to read from
filename = "data/output-info.txt"

# Suck up the data
with open(filename) as f:
    lines = f.readlines()

# Process it into nice lines
data = []
for line in lines[1:]:
    if line:
        fn, phi0, phi0dot = line.strip().split("\t")
        phi0 = float(phi0)
        phi0dot = float(phi0dot)
        data.append((fn, phi0, phi0dot))


# Read the data files for each run in the sweep
for file, phi0, phi0dot in data:
    results = load_data(file)
    details = analyze(results["a"], results["adot"], results["addot"])
    #print(details["efolds"])
    #store in runs
    #
    runs.a.append(results["a"][0])
    runs.phi.append(results["phi0"][0])
    runs.phidot.append(results["phi0dot"][0])
    #
    runs.H.append(results["H"][0])
    runs.rho.append(results["rho"][0])
    runs.drho2.append(results["deltarho2"][0])
    runs.hpotential.append(results["phi2pt"][0])
    #
    if "efolds" in details:
        runs.Nef.append(details["efolds"])
    else:
        details["efolds"] = 0.0
        runs.Nef.append(details["efolds"])
    #
    if details["efolds"] >= Nef_crit:
        runs.infl.append(1)
    else:
        runs.infl.append(0)

    #epsilon = slow_roll_epsilon(results["a"], results["adot"], results["addot"])
    #plt.plot(results["t"], epsilon, 'b-')
    #plt.show()



#write data to file
sep = ", "
fout = open(filename_swp, "w")
for i in range(0,len(runs.phi)):
    fout.write(str(runs.a[i]) + sep + str(runs.phi[i]) + sep + str(runs.phidot[i]) + sep + str(runs.H[i]) + sep + str(runs.rho[i]) + sep + str(runs.drho2[i]) + sep + str(runs.hpotential[i]) + sep + str(runs.Nef[i]) + sep + str(runs.infl[i]) + "\n")
fout.close()



#make plots in pdf file
#####
pdf_pages = PdfPages(sweep_plt)
#
fig = plt.figure(figsize=(7.0,7.0),dpi=100)
phase_plotter(np.asarray(runs.phi), np.asarray(runs.phidot), np.asarray(runs.Nef), np.asarray(runs.infl), '$N_{ef}$')
pdf_pages.savefig(fig)
#
fig = plt.figure(figsize=(7.0,7.0),dpi=100)
phase_plotter(np.asarray(runs.phi), np.asarray(runs.phidot), np.asarray(runs.rho)+np.asarray(runs.drho2), np.asarray(runs.infl), '$\\rho$ + $\delta \\rho^{2}$')
pdf_pages.savefig(fig)
#
fig = plt.figure(figsize=(7.0,7.0),dpi=100)
phase_plotter(np.asarray(runs.phi), np.asarray(runs.phidot), np.asarray(runs.drho2)/np.asarray(runs.rho), np.asarray(runs.infl), '$\delta \\rho^{2}$/$\\rho$')
pdf_pages.savefig(fig)
#
fig = plt.figure(figsize=(7.0,7.0),dpi=100)
phase_plotter(np.asarray(runs.phi), np.asarray(runs.phidot), np.asarray(runs.hpotential)/(np.asarray(runs.H)*np.asarray(runs.H)/(2*2*np.pi*np.pi)), np.asarray(runs.infl), '$<(\\delta\\phi)^{2}>_{t_{0}} / \\frac{H_{0}^{2}}{4\\pi^{2}} $')
pdf_pages.savefig(fig)
#
pdf_pages.close()





