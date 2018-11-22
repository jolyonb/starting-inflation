"""
model

Describes an inflationary model
"""

class NotImplementedError(Exception):
    """Raised when a required routine is not implemented in a subclass"""

class InflationModel(object):
    def __init__(self, **kwargs):
        """
        Initialize the model
        """
        raise NotImplementedError()

    def potential(self, phi0):
        """
        Value of the potential given background field value phi0
        """
        raise NotImplementedError()

    def dpotential(self, phi0):
        """
        Value of the derivative of the potential given background field value phi0
        """
        raise NotImplementedError()

    def ddpotential(self, phi0):
        """
        Value of the second derivative of the potential given background field value phi0
        """
        raise NotImplementedError()

    def dddpotential(self, phi0):
        """
        Value of the third derivative of the potential given background field value phi0
        """
        raise NotImplementedError()

    def ddddpotential(self, phi0):
        """
        Value of the fourth derivative of the potential given background field value phi0
        """
        raise NotImplementedError()

class LambdaPhi4(InflationModel):
    def __init__(self, **kwargs):
        self.lamda = kwargs['lamda']

    def potential(self, phi0):
        """
        Value of the potential given background field value phi0
        """
        return self.lamda * phi0**4 / 4

    def dpotential(self, phi0):
        """
        Value of the derivative of the potential given background field value phi0
        """
        return self.lamda * phi0**3

    def ddpotential(self, phi0):
        """
        Value of the second derivative of the potential given background field value phi0
        """
        return self.lamda * 3 * phi0 * phi0

    def dddpotential(self, phi0):
        """
        Value of the third derivative of the potential given background field value phi0
        """
        return self.lamda * 6 * phi0

    def ddddpotential(self, phi0):
        """
        Value of the fourth derivative of the potential given background field value phi0
        """
        return self.lamda * 6
