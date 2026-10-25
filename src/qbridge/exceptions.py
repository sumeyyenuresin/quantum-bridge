"""Exceptions raised by Quantum-Bridge."""


class QBridgeError(Exception):
    """Base class for all Quantum-Bridge errors."""


class UnsupportedOperationError(QBridgeError):
    """An instruction cannot be translated without losing information.

    Quantum-Bridge never silently drops or approximates an operation: anything
    that cannot be mapped exactly (classical control flow, mid-circuit
    measurement, reset, ...) raises this error instead.
    """


class ParameterBindingError(QBridgeError):
    """The values passed to a converted circuit do not match its parameters."""
