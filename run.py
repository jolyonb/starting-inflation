#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py

Performs an evolution
"""
import time
from math import sqrt
from evolver.integrator import Driver, Status
from evolver.initialize import create_package, create_parameters
from evolver.inflation import LambdaPhi4
from evolver.model import Model

# Initialize all settings
lamda = 1e-9
filename = "data/output"
package = create_package(phi0=22.5,
                         phi0dot=-0.01,
                         infmodel=LambdaPhi4(lamda=lamda),
                         end_time=5000 * sqrt(1e-6/lamda),
                         basefilename=filename,
                         hartree=False,
                         perturbBD=True,
                         timestepinfo=[200, 10])  # ~steps per efold (inside horizon),
                                                  # ~steps per efold (outside horizon)
parameters = create_parameters(package)

# Create the model
model = Model(parameters)
model.save(filename + ".params")

# Construct the driver
driver = Driver(model)

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
