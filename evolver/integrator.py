# -*- coding: utf-8 -*-
"""
Integration library built on top of scipy.integrate.ode

Uses a Driver class and a Model class, built on top of the AbstractModel class

This library is agnostic as to the model, which must be implemented as a subclass of
AbstractModel
"""
from enum import Enum
import pickle
from scipy.integrate import ode


class IntegrationError(Exception):
    """Raised when an integration error arises"""


class Status(Enum):
    """Status of Driver object"""
    Initializing = 0
    OK = 1
    Finished = 2
    IntegrationError = -1
    Terminated = -2


class AbstractModel(object):
    """Abstract class that handles all model and run-depdendent information"""

    def __init__(self, run_params):
        """
        Sets up all data for an evolution

        Arguments:
            * run_params: Dictionary of all initialization information.

        run_params can include whatever keys you like. The following keys have
        meaning for AbstractModel:
            * initial_data: The initial data for the integration (required)
            * start_time: The initial value for time t (default 0)
            * end_time: The maximum value of time t to integrate to (default 100)
            * atol: Absolute tolerance in integration (default 1e-10)
            * rtol: Relative tolerance in integration (default 1e-10)
            * separator: Separator to use between fields when outputting data (default ", ")
        """
        # Defaults
        self.parameters = {
            'start_time': 0,
            'end_time': 100,
            'atol': 1e-10,
            'rtol': 1e-10,
            'separator': ', '
        }

        # Add in new information
        self.parameters.update(run_params)

        # Store values that will be used repeatedly
        self.initial_data = self.parameters['initial_data']
        self.start_time = self.parameters['start_time']
        self.end_time = self.parameters['end_time']
        self.atol = self.parameters['atol']
        self.rtol = self.parameters['rtol']
        self.separator = self.parameters['separator']

        # Initialize internal flags
        self.halt = False
        self.haltmsg = None

    def begin(self):
        """A function that is called before evolution begins"""
        raise NotImplementedError()

    def cleanup(self):
        """A function that is called after evolution finishes"""
        raise NotImplementedError()

    def derivatives(self, time, data):
        """Computes derivatives for evolution"""
        raise NotImplementedError()

    def compute_timestep(self, time, data):
        """Compute the desired timestep at this point in the evolution"""
        raise NotImplementedError()

    def solout(self, time, data):
        """
        A function that is called after every internal timestep.
        If this returns -1, it terminates the evolution.
        """
        if self.halt:
            return -1
        return 0

    def write_data(self, time, data):
        """
        A function that is called between timesteps (and also at the
        start/end of integration), when data is ready for writing.
        """
        raise NotImplementedError()

    def format_data(self, time, data):
        """Format the current data for output"""
        return (str(time) + self.separator
                + self.separator.join(map(str, data)) + "\n")

    def save(self, filename):
        """Save this model to file (using pickle)"""
        with open(filename, 'wb') as f:
            pickle.dump(self.parameters, f)

    @classmethod
    def load(cls, filename):
        """Instantiate a model from a previously-saved file"""
        with open(filename) as f:
            data = pickle.load(f)
        return cls(data)


class Driver(object):
    """Sets up and runs the evolution"""

    def __init__(self, model):
        """
        Set parameters for driving the evolution

        Arguments:
            * model: An AbstractModel subclass instance
        """
        self.status = Status.Initializing
        self.error_msg = ""
        self.model = model

        # Initialize the integrator
        self.data = model.initial_data
        self.integrator = ode(model.derivatives)
        self.integrator.set_integrator('dopri5',
                                       nsteps=10000,
                                       rtol=model.rtol,
                                       atol=model.atol)
        self.integrator.set_initial_value(model.initial_data, model.start_time)
        self.integrator.set_solout(model.solout)

        self.status = Status.OK

    @property
    def time(self):
        """Returns the current time that the integrator is at"""
        return self.integrator.t

    def run(self):
        """
        Runs the evolution

        The resulting status is stored in self.status.
        If it stores Status.IntegrationError, the error message
        is available through self.error_msg.
        """
        # Make sure we're ready to roll
        if self.status != Status.OK:
            raise ValueError("Cannot begin evolution as status is not OK.")

        # Initialization
        self.model.begin()

        # Write initial data
        self.model.write_data(self.time, self.data)

        # Integration loop
        newtime = self.time
        while True:
            # Check to see if we're finished
            if newtime >= self.model.end_time:
                self.status = Status.Finished
                break
            elif self.model.halt:
                self.status = Status.Terminated
                self.error_msg = self.model.haltmsg
                break

            # Construct the time to integrate to
            newtime = self.time + self.model.compute_timestep(self.time, self.data)

            # Take a step to newtime
            try:
                results = self.integrator.integrate(newtime, relax=True)
                if not self.integrator.successful():
                    raise IntegrationError("DOPRI Error Code {}"
                                           .format(self.integrator.get_return_code()))
                self.data = results
            except IntegrationError as e:
                self.status = Status.IntegrationError
                self.error_msg = e.args[0]
                break

            # Write the data
            self.model.write_data(self.time, self.data)

        # Clean up
        self.model.cleanup()
