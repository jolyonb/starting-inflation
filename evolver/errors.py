"""
errors

Contains all error classes for the module
"""

class IntegrationError(Exception):
    """Raised when an integration error arises"""


class TerminateError(Exception):
    """Raised when integration can no longer continue"""
