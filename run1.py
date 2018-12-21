#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py

Performs a modified run of a previously generated data set
ex. Hartree off -> on
"""
import time
from math import sqrt
from evolver.integrator import Driver, Status
from evolver.initialize import create_package, create_parameters
from evolver.inflation import LambdaPhi4
from evolver.model import Model
import argparse

parser = argparse.ArgumentParser(description="load data from previous run")
parser.add_argument("filename", help="Base of the filename to read data in from")
args = parser.parse_args()

# Create the model
model = Model.load(args.filename + ".params")

# modify the parameters of a previous run for the current run
model.parameters['hartree'] = True
model.eomparams.hartree = True

# model.parameters['basefilename'] = 'newfile'
# model.basefilename = 'newfile'

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
