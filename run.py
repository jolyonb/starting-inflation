#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performs an evolution
"""
import time
import sys
from integrator import Driver, Status
from initialize import make_initial_data, Model

import numpy as np

# Construct parameters and initial data
debug = False
lamda = 1e-9
phi0 = 30.0
phi0dot = -0.005
Hback0 = np.sqrt((0.5*phi0dot*phi0dot + (1/4)*lamda*phi0**(4))/3)
Rmax = 2.0/Hback0
k_modes = 2
kappa = 20.0*Hback0
hartree = True
filename = "output.dat"
filename2 = "output2.dat"
perturbed_ratio = 0.1

###### debug 20. Nov 2018 to check why Bunch-Davies is slightly off a.c.t. Mathematica nb
print ("Hback0:", Hback0)
print ("Rmax:", Rmax)
print ("kappa:", kappa)
###### end debug

randomize = False
seed = None
solution = 1
params, initial_data = make_initial_data(phi0, phi0dot, Rmax, k_modes, hartree,
                                         lamda, kappa, perturbed_ratio, randomize, seed, solution, filename, filename2)

# sys.exit()

# Specify timing information
start_time = 0
end_time = 5000*np.sqrt(1e-6/lamda)
timestep = 0.5*np.sqrt(1e-6/lamda)

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
