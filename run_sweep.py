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
filename = "data/large_Hoff_sweep"
hartree = True
if hartree:
    num_modes = 40
else:
    num_modes = 2

# Background fields
# Note that a step of 1 only does the start value
phi0start = 25
phi0stop = 32
phi0steps = 1
phi0dotstart = -0.1
phi0dotstop = 0.1
phi0dotsteps = 1
# Number of runs to perform at each step
numruns = 20

# Construct our steps
phi0s = np.linspace(phi0start, phi0stop, phi0steps)
phi0dots = np.linspace(phi0dotstart, phi0dotstop, phi0dotsteps)
if not hartree:
    numruns = 1
runnums = [i + 1 for i in range(numruns)]

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
                         hartree=hartree,
                         perturbBD=True,
                         timestepinfo=[200, 10],
                         num_k_modes=num_modes)

for x, y, runnum in tqdm(list(itertools.product(phi0s, phi0dots, runnums))):
    # Construct the filename
    run += 1
    fn = filename + "-{}".format(run)
    infofile.write("{}\t{}\t{}\n".format(fn, x, y))

    # Update package
    package['phi0'] = x
    package['phi0dot'] = y
    package['basefilename'] = fn

    parameters = create_parameters(package)

    # Create the model
    model = Model(parameters)
    model.save(fn + ".params")

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

infofile.close()

end = time.time()
print("Finished in {} s".format(round(end - start, 4)))
