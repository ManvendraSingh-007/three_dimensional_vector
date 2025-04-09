"""
vector_errors.py

This module defines custom exception classes for the Vector library. These exceptions are designed to handle errors
specific to vector operations, providing meaningful error messages for issues such as invalid operations, out-of-bounds
access, and unsupported operations on vectors. By centralizing error handling in a dedicated file, the Vector library
can manage and extend error handling consistently and clearly throughout the codebase.

Classes:
    - VectorError: Base class for all vector-related exceptions.
    - DimensionMismatchError: Raised when operations are attempted on vectors of different dimensions.
    - ZeroVectorError: Raised when operations that require non-zero vectors are attempted on zero vectors.
    - IndexError: Raised when attempting to access a vector component with an invalid index.
    - InvalidOperationError: Raised when an unsupported operation is attempted on vectors.

Usage:
    To use these exceptions, import them into your vector module and raise them as needed when handling errors
    specific to vector operations.
"""

class VectorError(Exception):
    """Base class for all vector-related exceptions."""
    pass

class DimensionMismatchError(VectorError):
    """Raised when operations are attempted on vectors of different dimensions."""
    def __init__(self, message="Dimension mismatch"):
        super().__init__(message)
        

class ZeroVectorError(VectorError):
    """Raised when operations that require non-zero vectors are attempted on zero vectors."""
    pass

class IndexError(VectorError):
    """Raised when attempting to access a vector component with an invalid index."""
    pass

class InvalidOperationError(VectorError):
    """Raised when an unsupported operation is attempted on vectors."""
    pass

