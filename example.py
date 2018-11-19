#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolves a simple harmonic oscillator with period 5 through 10 periods as an
example of how to use the integrator library. Note that this is more complicated
than need be for this simple example, but it demonstrates the tools that will become
crucial for runs with more fields.

Try running this, then running plot.py to see the results.
"""
from math import pi
import numpy as np
from integrator import AbstractModel, Driver, Status

class Parameters(object):
    def __init__(self, period):
        self.period = period
        self.omega = 2 * pi / period

class Model(AbstractModel):
    def derivatives(self, time, data):
        """
        Computes derivatives for evolution

        Arguments:
            * time: The current time
            * data: The current data as a numpy array

        Returns:
            * derivatives: The derivatives given the current data and time. Must be the
                           same size numpy array as data.
        """
        # Obtain any parameters we need
        omega = self.parameters.omega

        # Unpack the data
        x, v = unpack_value(data, 0)

        # First derivatives can be straightforwardly constructed
        firstderivs = first_derivatives(data)

        # Second derivatives need the equations of motion
        secondderivs = np.array([-omega**2 * x])

        # Combine first and second derivatives and return the result
        derivs = np.concatenate((firstderivs, secondderivs))
        return derivs

def pack_values(*args):
    """
    Take in an unspecified number of arguments as (value, deriv) tuples,
    and pack them into a numpy array
    """
    values = np.array([val[0] for val in args])
    derivs = np.array([val[1] for val in args])
    return np.concatenate((values, derivs))

def unpack_value(data, item_number):
    """
    Take in a numpy array as date, and extract the (value, deriv) pair for
    the given item_number
    """
    return data[item_number], data[int(item_number + data.size / 2)]

def first_derivatives(data):
    """
    Returns a numpy array of the time derivatives of the first half of the data
    (which is conveniently given by the second half of the data)
    """
    return data[int(data.size/2):]


# Set up the output file
f = open("output.dat", "w")

# Specify initial data as a value and initial time derivative
initial = (1, 0)
# Pack all initial data into an array
initial_data = pack_values(initial)

# Specify parameters
start_time = 0
end_time = 10
timestep = 0.1
period = 5
params = Parameters(period)

# Construct the driver
debug = True
driver = Driver(Model, initial_data, params, start_time, end_time, timestep, debug=debug)

# Perform the evolution
print("Beginning evolution")
driver.run(f)

# Check to see what our status is
if driver.status == Status.IntegrationError:
    print("Unable to integrate further: {}".format(driver.msg))
elif driver.status == Status.Finished:
    print("Evolution completed!")

# Tidy up
f.close()
print("Done!")
