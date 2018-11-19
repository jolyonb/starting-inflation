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
lamda = 1e-6
phi0 = 25.0
phi0dot = 0.001
#define a couple parameters in terms of H_{0} (which we approximate by its background value here -- Hback0 -- the true H_{0} will have a small correction from this due to the inclusion of \delta \rho2)
Hback0 = np.sqrt((1.0/3.0)*(0.5*phi0dot*phi0dot + (1.0/4.0)*lamda*phi0**(4.0)))
print("H0 (for R and \kappa): {0}".format(Hback0))
Rmax = 2.0/Hback0
k_modes = 40
kappa = 20.0*Hback0
hartree = True
filename = "output.dat"
filename2 = "output2.dat"

perturbed_ratio = 0.1

randomize = False
seed = None
solution = 1

params, initial_data = make_initial_data(phi0, phi0dot, Rmax, k_modes, hartree,
                                         lamda, kappa, perturbed_ratio, randomize, seed, solution, filename, filename2)

# sys.exit()

# Specify timing information
start_time = 0
end_time = 5000*np.sqrt(1e-6/lamda)
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
