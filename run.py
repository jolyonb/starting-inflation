#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs an evolution
"""
import time
from math import sqrt
from evolver.integrator import Driver, Status
from evolver.initialize import make_initial_data, Model
from evolver.model import LambdaPhi4

# Initialize all settings
debug = False
hartree = True
k_modes = 40
l1modeson = True
performrun = True
filename = "data/output"

# Inflation model
infmodel = LambdaPhi4(lamda=1e-9)

# Background fields
phi0 = 25.0
phi0dot = 0.01

# Specify timing information
start_time = 0
end_time = 5000 * sqrt(1e-6/infmodel.lamda)
timestep = 0.05 * sqrt(1e-6/infmodel.lamda)

# Construct parameters class and initial data
params, initial_data = make_initial_data(phi0, phi0dot, k_modes, hartree, infmodel,
                                         filename, l1modeson=l1modeson)

# Perform the run
if __name__ == "__main__":
    # Do we run?
    if not performrun:
        end_time = 0

    # Construct the driver
    driver = Driver(Model, initial_data, params, start_time, end_time, timestep, debug=debug)

    # Perform the evolution
    print("Beginning evolution")

    start = time.time()
    driver.run()
    end = time.time()

    print("Finished in {} s".format(round(end - start, 4)))

    # Check to see what our status is
    if driver.status == Status.IntegrationError:
        print("Unable to integrate further: {}".format(driver.error_msg))
    elif driver.status == Status.Terminated:
        print("Evolution completed with message: {}".format(driver.error_msg))
    elif driver.status == Status.Finished:
        print("Evolution completed!")
