#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs an evolution
"""
import time
import sys
from integrator import Driver, Status
from initialize import make_initial_data, Model

# Construct parameters and initial data
debug = False
Rmax = 3.0
k_modes = 6
hartree = True
lamda = 1e-9
kappa = 3.0
phi0 = 30.0
phi0dot = 1.0
filename = "output.dat"
filename2 = "output2.dat"
perturbed_ratio = 0.1
randomize = True
seed = None
solution = 1
params, initial_data = make_initial_data(phi0, phi0dot, Rmax, k_modes, hartree,
                                         lamda, kappa, perturbed_ratio, randomize, seed,
                                         solution, filename, filename2)

sys.exit()

# Specify timing information
start_time = 0
end_time = 3000
timestep = 0.5

# Perform the run
if __name__ == "__main__":
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
    elif driver.status == Status.Finished:
        print("Evolution completed!")
