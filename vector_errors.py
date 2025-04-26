"""
vector_errors.py

This module defines custom exception classes for the Vector library. These exceptions are designed to handle errors
specific to vector operations, providing meaningful error messages for issues such as invalid operations, out-of-bounds
access, and unsupported operations on vectors. By centralizing error handling in a dedicated file, the Vector library
can manage and extend error handling consistently and clearly throughout the codebase.

Classes:
    - VectorError: Base class for all vector-related exceptions.
    - ZeroVectorError: Raised when an operation cannot be performed on a zero vector
    - VectorDimensionError: Raised when operations are attempted on vectors of different dimensions
    - VectorOperationError: Raised when an invalid vector operation is attempted
    - VectorNormalizationError: Raised when vector normalization fails
    - VectorAngleError: Raised when angle calculation between vectors fails
    - VectorIndexError: Raised when an invalid vector index is accessed
    - VectorDivisionError: Raised when attempting to divide a vector by zero
    - VectorTypeError: Raised when an operation is performed with incompatible types
    - VectorInputError: Raised when invalid input is provided for vector creation

Usage:
    To use these exceptions, import them into your vector module and raise them as needed when handling errors
    specific to vector operations.
"""

class VectorError(Exception):
    """Base class for all vector-related errors"""
    pass

class ZeroVectorError(VectorError):
    """Raised when an operation cannot be performed on a zero vector"""
    def __init__(self, operation: str):
        super().__init__(f"Cannot {operation} a zero vector")

class VectorDimensionError(VectorError):
    """Raised when vector dimensions are incorrect"""
    def __init__(self, expected: int, actual: int):
        super().__init__(f"Expected {expected} dimensions, got {actual}")

class VectorOperationError(VectorError):
    """Raised when an invalid vector operation is attempted"""
    def __init__(self, operation: str, reason: str):
        super().__init__(f"Invalid vector operation '{operation}': {reason}")

class VectorNormalizationError(VectorError):
    """Raised when vector normalization fails"""
    def __init__(self, message, magnitude):
        super().__init__(f"{message}, {magnitude}")

class VectorAngleError(VectorError):
    """Raised when angle calculation between vectors fails"""
    def __init__(self, message, v1, v2):
        super().__init__(message, v1, v2)

class VectorIndexError(VectorError, IndexError):
    """Raised when an invalid vector index is accessed"""
    def __init__(self, index: int):
        super().__init__(f"Vector index out of range: {index} (must be 0, 1, or 2)")

class VectorDivisionError(VectorError, ZeroDivisionError):
    """Raised when attempting to divide a vector by zero"""
    def __init__(self):
        super().__init__("Cannot divide vector by zero")

class VectorTypeError(VectorError, TypeError):
    """Raised when an operation is performed with incompatible types"""
    def __init__(self, operation: str, expected_type: str, actual_type: str):
        super().__init__(
            f"Cannot perform '{operation}' between Vector and {actual_type}, "
            f"expected {expected_type}"
        )

class VectorInputError(VectorError, ValueError):
    """Raised when invalid input is provided for vector creation"""
    def __init__(self, input_type: str, details: str):
        super().__init__(f"Invalid {input_type} for vector creation: {details}")

class VectorInitializationError(VectorError):
    """Raised during invalid Vector initilization"""
    def __init__(self, operation: str):
        super().__init__(f"Cannot initalize Vector, {operation}")