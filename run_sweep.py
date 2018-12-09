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
filename = "data/Hoff_sweep"
hartree = False
if hartree:
    num_modes = 40
else:
    num_modes = 2

# Background fields
phi0s = np.linspace(22, 32, 10)
# Minimum phi0dot should be around -0.025, max should be around 0.025
phi0dots = np.linspace(-0.01, 0.01, 10)

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
                         timestepinfo=[200, 10],
                         num_k_modes=num_modes)

for x, y in tqdm(list(itertools.product(phi0s, phi0dots))):
    # Construct the filename
    run += 1
    fn = filename + "-{}".format(run)
    infofile.write("{}\t{}\t{}\n".format(fn, x, y))

    #Update package
    package['phi0'] = x
    package['phi0dot'] = y
    package['basefilename'] = fn

    parameters = create_parameters(package)

    #Create the model
    model = Model(parameters)
    model.save(fn + ".params")

    #Construct the driver
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
