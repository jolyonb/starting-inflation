#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs a sweep over parameter space (evolutions only)
"""
import os
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
lamda = 1e-10
settings = {"off": False, "bunchdavies": False, "hartree": 1}
filename = "data/plots"

# Split the filename into a directory and a filename
directory, filename = os.path.split(filename)
os.chdir(directory)

# Background fields
# Note that a step of 1 only does the start value
phi0start = 20
phi0stop = 30
phi0steps = 3
phi0dotstart = -0.1
phi0dotstop = 0.1
phi0dotsteps = 3

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
                         perturbBD=False,
                         timestepinfo=[200, 10],
                         fulloutput=False)

def perform_run(phi0, phi0dot, filename, hartree, bunchdavies):
    # Update package
    package['phi0'] = phi0
    package['phi0dot'] = phi0dot
    package['basefilename'] = filename
    package['hartree'] = hartree
    package['perturbBD'] = not bunchdavies

    parameters = create_parameters(package)
    if parameters is None:
        return False

    # Create the model
    model = Model(parameters)
    model.save(filename + ".params")

    # Construct the driver
    driver = Driver(model)

    # Perform the evolution
    driver.run()

    # Check to see what our status is
    if driver.status == Status.IntegrationError:
        with open(filename + ".info", "a") as f:
            f.write("Unable to integrate further: {}\n".format(driver.error_msg))
    elif driver.status == Status.Terminated:
        with open(filename + ".info", "a") as f:
            f.write("Evolution completed with message: {}".format(driver.error_msg))

    return True

# Sweep through all the runs
for x, y in tqdm(list(itertools.product(phi0s, phi0dots))):
    run += 1
    fn = filename + "-{}".format(run)
    if settings["off"]:
        # Construct the filename
        fnoff = fn + "-off"
        if perform_run(x, y, fnoff, False, True):
            infofile.write("{}\t{}\t{}\n".format(fnoff, x, y))
    if settings["bunchdavies"]:
        # Construct the filename
        fnoff = fn + "-bd"
        if perform_run(x, y, fnoff, True, True):
            infofile.write("{}\t{}\t{}\n".format(fnoff, x, y))
    if settings["hartree"] > 0:
        for i in range(settings["hartree"]):
            # Construct the filename
            fnon = fn + "-" + str(i + 1)
            if perform_run(x, y, fnon, True, False):
                infofile.write("{}\t{}\t{}\n".format(fnon, x, y))

infofile.close()

end = time.time()
print("Finished in {} s".format(round(end - start, 4)))
