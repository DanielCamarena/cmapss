class ValidationError(Exception):
    """Raised when request payload is invalid for prediction."""


class ServiceUnavailableError(Exception):
    """Raised when prediction service path is temporarily unavailable."""

