#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs a sweep over parameter space (evolutions only)
"""
import time
from math import sqrt
import numpy as np
import itertools
from tqdm import tqdm
from evolver.integrator import Driver, Status
from evolver.initialize import create_package, create_parameters
from evolver.inflation import LambdaPhi4
from evolver.model import Model

# Initialize all settings
lamda = 1e-9
hartree_settings = [True, False]  # Make a list of Hartree settings here
filename = "data/large_sweep"

# Background fields
# Note that a step of 1 only does the start value
phi0start = 25
phi0stop = 32
phi0steps = 3
phi0dotstart = -0.05
phi0dotstop = 0.05
phi0dotsteps = 3

# Number of runs to perform at each step (for Hartree on)
numruns = 2

# Fix the number of modes
if True in hartree_settings:
    num_modes = 40
else:
    num_modes = 2

# Construct our steps
phi0s = np.linspace(phi0start, phi0stop, phi0steps)
phi0dots = np.linspace(phi0dotstart, phi0dotstop, phi0dotsteps)


run = 0
start = time.time()
print("Starting!")

infofile = open(filename + "-info.txt", "w")
infofile.write("filename\tphi0\tphi0dot\n")

package = create_package(phi0=None,
                         phi0dot=None,
                         infmodel=LambdaPhi4(lamda=lamda),
                         end_time=5000*sqrt(1e-6/lamda),
                         basefilename=None,
                         perturbBD=True,
                         timestepinfo=[200, 10],
                         num_k_modes=num_modes)

def perform_run(phi0, phi0dot, filename, hartree):
    # Update package
    package['phi0'] = phi0
    package['phi0dot'] = phi0dot
    package['basefilename'] = filename
    package['hartree'] = hartree

    parameters = create_parameters(package)

    # Create the model
    model = Model(parameters)
    model.save(filename + ".params")

    # Construct the driver
    driver = Driver(model)

    # Perform the evolution
    driver.run()

    # Check to see what our status is
    if driver.status == Status.IntegrationError:
        with open(fn + ".info", "a") as f:
            f.write("Unable to integrate further: {}\n".format(driver.error_msg))
    elif driver.status == Status.Terminated:
        with open(fn + ".info", "a") as f:
            f.write("Evolution completed with message: {}".format(driver.error_msg))

# Sweep through all the runs
for x, y in tqdm(list(itertools.product(phi0s, phi0dots))):
    run += 1
    fn = filename + "-{}".format(run)
    if False in hartree_settings:
        # Construct the filename
        fnoff = fn + "-off"
        infofile.write("{}\t{}\t{}\n".format(fnoff, x, y))
        perform_run(x, y, fnoff, False)
    if True in hartree_settings:
        for i in range(numruns):
            # Construct the filename
            fnon = fn + "-" + str(i + 1)
            infofile.write("{}\t{}\t{}\n".format(fnon, x, y))
            perform_run(x, y, fnon, True)

infofile.close()

end = time.time()
print("Finished in {} s".format(round(end - start, 4)))
