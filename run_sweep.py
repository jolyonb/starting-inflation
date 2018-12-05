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
from evolver.initialize import make_initial_data, Model
from evolver.model import LambdaPhi4

# Initialize all settings
debug = False
hartree = False
k_modes = 40
l1modeson = True
filename = "data/output"

# Inflation model
infmodel = LambdaPhi4(lamda=1e-9)

# Background fields
phi0 = np.linspace(20, 40, 10)
# Minimum phi0dot should be around -0.025, max should be around 0.025
# phi0dot = np.linspace(-0.025, 0.025, 3)
phi0dot = np.linspace(-0.01, 0.01, 10)

# Specify timing information
start_time = 0
end_time = 5000 * sqrt(1e-6/infmodel.lamda)
timestepinfo = [1000, 10]
# ~steps per efold (inside horizon), ~steps per efold (outside horizon)

run = 0
start = time.time()
print("Starting!")

infofile = open(filename + "-info.txt", "w")
infofile.write("filename\tphi0\tphi0dot\n")

for x, y in tqdm(list(itertools.product(phi0, phi0dot))):
    # Construct the filename
    run += 1
    fn = filename + "-{}".format(run)
    infofile.write("{}\t{}\t{}\n".format(fn, x, y))

    # Construct parameters class and initial data
    params, initial_data = make_initial_data(x, y, k_modes, hartree, infmodel,
                                             fn, l1modeson=l1modeson)

    # Construct the driver
    driver = Driver(Model, initial_data, params, start_time, end_time, timestepinfo, debug=debug)

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
