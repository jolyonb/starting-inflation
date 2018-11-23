#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration library built on top of scipy.integrate.ode

Uses a Driver class and a Model class, built on top of the AbstractModel class

This library is agnostic as to the model, which must be implemented as a subclass of
AbstractModel
"""
from enum import Enum
from scipy.integrate import ode
from evolver.errors import IntegrationError, TerminateError

class Status(Enum):
    """Status of Driver object"""
    Initializing = 0
    OK = 1
    Finished = 2
    IntegrationError = -1
    Terminated = -2


class AbstractModel(object):
    """Abstract class that handles integration"""

    def __init__(self, initial_data, parameters, start_time,
                 atol=1e-10, rtol=1e-10, separator=", ", debug=False):
        """
        Initialize the integrator

        Arguments:
            * initial_data: The initial data for the integration
            * parameters: Any parameters that the derivatives computation should have
                          access to
            * start_time: The initial value for time t
            * atol: Absolute tolerance in integration
            * rtol: Relative tolerance in integration
            * separator: Separator to use between fields when outputting data
            * debug: If True, requests debug output after each integration step

        Returns: None
        """
        # Store information
        self.data = initial_data
        self.parameters = parameters
        self.separator = separator
        self.debug = debug

        # Set up the integrator
        self.integrator = ode(self.derivatives).set_integrator('dopri5',
                                                               nsteps=10000,
                                                               rtol=rtol,
                                                               atol=atol)
        self.integrator.set_initial_value(initial_data, start_time)

    @property
    def time(self):
        """
        Returns the current time that the model is at.

        Returns: time
        """
        return self.integrator.t

    def step(self, newtime):
        """
        Takes a step forwards in time

        Arguments:
            * newtime: The time value that should be evolved to in this step

        Returns: None
        """
        # If debug mode is on, we step slowly through the integration
        if self.debug:
            # Initialize parameters
            stepcount = 0
            delta = 0

            # Take steps until we get to the desired time
            while self.time < newtime:
                # Take a step
                stepcount += 1
                last_time = self.time
                # step=True says to take a single step, which may overshoot newtime
                try:
                    results = self.integrator.integrate(newtime, step=True)
                except TerminateError:
                    raise
                if not self.integrator.successful():
                    raise IntegrationError("DOPRI Error Code {}"
                                           .format(self.integrator.get_return_code()))
                # Store the last timestep taken
                delta = self.time - last_time

            # Store result
            self.data = results

            # Reporting
            msg = "Integrator: Stepped to t = {} in {} steps, last stepsize was {}"
            print(msg.format(round(self.time, 5),
                             stepcount,
                             round(delta, 5)))
        else:
            # If we're not in debug mode, then let scipy have it's head
            # Take the full step
            # relax=True says to let the step overshoot the end time
            results = self.integrator.integrate(newtime, relax=True)
            if not self.integrator.successful():
                raise IntegrationError("DOPRI Error Code {}"
                                       .format(self.integrator.get_return_code()))

            # Store result
            self.data = results

    def write_data(self):
        """
        Writes the current data to file

        Returns: None
        """
        sep = self.separator
        self.parameters.f.write(str(self.time) + sep + sep.join(map(str, self.data)) + "\n")

    def derivatives(self, time, data):
        """
        Computes derivatives for evolution

        Arguments:
            * time: The current time
            * data: The current data as a numpy array

        Returns:
            * derivatives: The derivatives given the current data and time. Must be the
                           same size numpy array as data.

        Note that this method has access to self.parameters
        Must be implemented by a model
        """
        raise NotImplementedError()


class AbstractParameters(object):
    """Abstract class that handles file stuff"""

    def open_file(self):
        """
        Opens the file handle to write to

        Returns: None
        """
        self.f = open(self.filename, "w")
        self.f2 = open(self.filename2, "w")

    def close_file(self):
        """
        Closes the file handle

        Returns: None
        """
        self.f.close()
        self.f2.close()


class Driver(object):
    """
    Sets up and runs the evolution
    """

    def __init__(self, Model, initial_data, parameters, start_time, end_time, timestep,
                 atol=1e-10, rtol=1e-10, separator=", ", debug=False):
        """
        Set parameters for driving the evolution

        Arguments:
            * Model: The model class to use in integration, which should subclass AbstractModel
            * initial_data: The initial data for the integration
            * parameters: Any parameters that the derivatives computation should have
                          access to
            * start_time: The initial value for time t
            * end_time: The final value of time to evolve to
            * timestep: The timestep between which to output data
            * atol: Absolute tolerance in integration
            * rtol: Relative tolerance in integration
            * separator: Separator to use between fields when outputting data
            * debug: If True, requests debug output after each integration step

        Returns: None
        """
        self.status = Status.Initializing

        # Save parameters
        self.start_time = start_time
        self.end_time = end_time
        self.timestep = timestep
        self.debug = debug

        # Initialize the data objects
        self.data = Model(initial_data=initial_data,
                          parameters=parameters,
                          start_time=start_time,
                          atol=atol,
                          rtol=rtol,
                          separator=separator,
                          debug=debug)

        # Ready to roll
        self.error_msg = ""
        self.status = Status.OK

    def run(self):
        """
        Runs the evolution

        Returns: None

        The resulting status is stored in self.status. If it stores Status.IntegrationError,
        the error message is available through self.error_msg.
        """
        # Make sure we're ready to roll
        if self.status != Status.OK:
            raise ValueError("Cannot begin evolution as status is not OK.")

        # Write the initial conditions
        self.data.parameters.open_file()
        self.data.write_data()
        self.data.write_extra_data()

        # Integration loop
        newtime = self.data.time
        while True:
            # Check to see if we're finished
            if newtime >= self.end_time:
                self.status = Status.Finished
                self.data.parameters.close_file()
                return

            # Construct the time to integrate to
            # We try to keep the timing roughly the same between runs
            while newtime <= self.data.time:
                newtime += self.timestep
            if newtime > self.end_time:
                newtime = self.end_time

            # Take a step to newtime
            try:
                self.data.step(newtime)
            except IntegrationError as e:
                self.status = Status.IntegrationError
                self.error_msg = e.args[0]
                self.data.parameters.close_file()
                return
            except TerminateError as e:
                self.status = Status.Terminated
                self.error_msg = e.args[0]
                self.data.parameters.close_file()
                return

            # Write the data
            self.data.write_data()
            self.data.write_extra_data()
