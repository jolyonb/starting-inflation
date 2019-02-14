#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun.py

Loads data from a previous run and evolves it again
Allows initializations to be changed!
"""
import time
from evolver.integrator import Driver, Status
from evolver.model import Model
import argparse

desc = "Reruns a previous run, possibly with changed parameters"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument("filename", help="Parameters file to load (.params file)")
args = parser.parse_args()

# Load the model
model = Model.load(args.filename + ".params")


#
# If you want to modify any parameters, do it here
#

model.parameters['fulloutput'] = True
model.fulloutput = True

# Change the filename
filename = "data/newrun"
model.parameters['basefilename'] = filename
model.basefilename = filename
model.save(filename + ".params")


#
# Now do the run again
#

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
