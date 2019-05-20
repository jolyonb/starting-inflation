#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs a series of runs at varying outer radius in order to demonstrate convergence
"""
import os
import time
from math import sqrt
import numpy as np
from evolver.integrator import Driver, Status
from evolver.initialize import create_package, create_parameters
from evolver.inflation import LambdaPhi4
from evolver.model import Model

# Initialize all settings
filename = "data/converge"

lamda = 1e-9
phi0 = 25
phi0dot = 0.05
k_max_factor = 3.5

# Set the outer radius to run with
radii = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

# Split the filename into a directory and a filename
directory, filename = os.path.split(filename)
os.chdir(directory)

run = 0
start = time.time()
print("Starting!")

infofile = open(filename + "-info.txt", "w")
infofile.write("filename\tphi0\tphi0dot\tR_max\n")

package = create_package(phi0=phi0,
                         phi0dot=phi0dot,
                         infmodel=LambdaPhi4(lamda=lamda),
                         end_time=5000*sqrt(1e-6/lamda),
                         k_max_factor=k_max_factor,
                         basefilename=None,
                         perturbBD=False,
                         timestepinfo=[200, 10],
                         num_k_modes=None,
                         fulloutput=True,
                         hartree=True)

# Sweep through all the runs
for Rval in radii:
    print(f"Running R of {Rval}")
    run += 1
    fn = filename + f"-{run}"

    # Update the model details for this run
    package['basefilename'] = fn
    package['Rmaxfactor'] = Rval

    # Run the model
    parameters = create_parameters(package)
    if parameters is None:
        print("Unable to construct initial conditions")
        continue
    model = Model(parameters)
    model.save(fn + ".params")
    driver = Driver(model)
    driver.run()

    # Check to see what our status is
    infofile.write(f"{fn}\t{phi0}\t{phi0dot}\t{Rval}\n")
    if driver.status == Status.IntegrationError:
        with open(fn + ".info", "a") as f:
            f.write("Unable to integrate further: {}\n".format(driver.error_msg))
    elif driver.status == Status.Terminated:
        with open(fn + ".info", "a") as f:
            f.write("Evolution completed with message: {}".format(driver.error_msg))

infofile.close()

end = time.time()
print("Finished in {} s".format(round(end - start, 4)))
