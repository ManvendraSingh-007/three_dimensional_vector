import math
from functools import total_ordering
from typing import Any, Tuple, List, Union, Optional, overload, Iterable, Sequence, Type, TypeVar
import numpy as np
from vector_errors import * # Default tolerance for floating-point comparisons
DEFAULT_ATOL = 1e-8

@total_ordering
class Vector:
    """
    Represents a 3-dimensional Euclidean vector (x, y, z).

    Supports common vector operations, operator overloading, and comparisons.
    Uses numpy for underlying calculations and caches magnitude for efficiency.
    """
    
    # Using __slots__ can slightly improve memory usage and attribute access speed
    # It prevents the creation of __dict__ for each instance.
    # Note: This means you cannot add arbitrary attributes to instances later.
    __slots__ = ('_components', '_magnitude')

    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, x: float) -> None: ...
    @overload
    def __init__(self, x: float, y: float) -> None: ...
    @overload
    def __init__(self, x: float, y: float, z: float) -> None: ...
    @overload
    def __init__(self, components: Union[List[float], Tuple[float, ...], np.ndarray]) -> None: ...
        
    def __init__(self, *args: Any) -> None:
        """
        Initialize a Vector object.

        Can be initialized in several ways:
        - Vector(): Creates a zero vector (0, 0, 0).
        - Vector(x): Creates a vector (x, 0, 0).
        - Vector(x, y): Creates a vector (x, y, 0).
        - Vector(x, y, z): Creates a vector (x, y, z).
        - Vector([x, y, z]) or Vector((x, y, z)): Creates a vector from a list or tuple.
        - Vector(numpy_array): Creates a vector from a 3-element numpy array.

        Parameters:
        *args: Arguments for initialization (see above).

        Raises:
        VectorInitializationError: If arguments are invalid or incompatible.
        """
        if len(args) == 0:
            self._components = np.array([0.0, 0.0, 0.0], dtype=float)
        elif len(args) == 1:
            arg = args[0]
            if isinstance(arg, (int, float)):
                # Vector(x) -> (x, 0, 0)
                self._components = np.array([float(arg), 0.0, 0.0], dtype=float)
            elif isinstance(arg, (list, tuple)):
                if len(arg) == 3:
                    self._components = np.array(arg, dtype=float)
                elif len(arg) == 2: # Allow list/tuple init for 2D
                    self._components = np.array([arg[0], arg[1], 0.0], dtype=float)
                elif len(arg) == 1: # Allow list/tuple init for 1D
                     self._components = np.array([arg[0], 0.0, 0.0], dtype=float)
                else:
                    raise VectorInitializationError(f"List or tuple argument must have 1, 2, or 3 elements, got {len(arg)}")
            elif isinstance(arg, np.ndarray):
                if arg.shape == (3,):
                    self._components = arg.astype(float) # Ensure float type
                else:
                     raise VectorInitializationError(f"Numpy array argument must have shape (3,), got {arg.shape}")
            elif isinstance(arg, Vector): # Allow copy constructor
                self._components = arg._components.copy()
            else:
                 raise VectorInitializationError(f"Single argument must be a number, list, tuple, numpy array, or Vector, got {type(arg).__name__}")
        elif len(args) == 2:
            # Vector(x, y) -> (x, y, 0)
            self._components = np.array([float(args[0]), float(args[1]), 0.0], dtype=float)
        elif len(args) == 3:
             # Vector(x, y, z) -> (x, y, z)
            self._components = np.array([float(args[0]), float(args[1]), float(args[2])], dtype=float)
        else: # len(args) > 3
            raise VectorInitializationError(f"Expected 0, 1, 2, or 3 scalar arguments, or a single sequence/array argument, got {len(args)} arguments.")
            
        self._magnitude: Optional[float] = None # Cache for magnitude

    # --- Properties for Coordinates ---
    @property
    def x(self) -> float:
        """Get the x-component of the vector."""
        return self._components[0]
    
    @x.setter
    def x(self, value: float) -> None:
        """Set the x-component and invalidate cached magnitude."""
        new_val = float(value)
        if self._components[0] != new_val:
            self._components[0] = new_val
            self.invalidate_cache()

    @property
    def y(self) -> float:
        """Get the y-component of the vector."""
        return self._components[1]
    
    @y.setter
    def y(self, value: float) -> None:
        """Set the y-component and invalidate cached magnitude."""
        new_val = float(value)
        if self._components[1] != new_val:
            self._components[1] = new_val
            self.invalidate_cache()

    @property
    def z(self) -> float:
        """Get the z-component of the vector."""
        return self._components[2]
    
    @z.setter
    def z(self, value: float) -> None:
        """Set the z-component and invalidate cached magnitude."""
        new_val = float(value)
        if self._components[2] != new_val:
            self._components[2] = new_val
            self.invalidate_cache()

    def invalidate_cache(self) -> None:
        """Reset the cached magnitude. Called when components change."""
        self._magnitude = None

    # --- Class Methods for Common Vectors ---
    @classmethod
    def zero(cls) -> 'Vector':
        """Return the zero vector: Vector(0, 0, 0)."""
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def ones(cls) -> 'Vector':
        """Return the vector with all components as one: Vector(1, 1, 1)."""
        return cls(1.0, 1.0, 1.0)
        
    @classmethod
    def unit_x(cls) -> 'Vector':
        """Return the unit vector along the x-axis: Vector(1, 0, 0)."""
        return cls(1.0, 0.0, 0.0)
        
    @classmethod
    def unit_y(cls) -> 'Vector':
        """Return the unit vector along the y-axis: Vector(0, 1, 0)."""
        return cls(0.0, 1.0, 0.0)
        
    @classmethod
    def unit_z(cls) -> 'Vector':
        """Return the unit vector along the z-axis: Vector(0, 0, 1)."""
        return cls(0.0, 0.0, 1.0)

    # --- Core Vector Operations ---
    def magnitude(self) -> float:
        r"""
        Calculate the magnitude (Euclidean norm or length) of the vector.

        Formula: $||\vec{v}|| = \sqrt{x^2 + y^2 + z^2}$

        Returns:
        float: The magnitude of the vector.
        """
        if self._magnitude is None: # Calculate magnitude if not cached
            self._magnitude = np.linalg.norm(self._components)
        return self._magnitude
        
    def magnitude_squared(self) -> float:
        """
        Calculate the squared magnitude of the vector.
        Often more efficient than magnitude() if only comparing lengths.

        Formula: $||\vec{v}||^2 = x^2 + y^2 + z^2$
        
        Returns:
            float: The squared magnitude of the vector.
        """
        # Avoids recalculating if magnitude is cached, otherwise avoids sqrt
        if self._magnitude is not None:
            return self._magnitude ** 2
        # Recompute without sqrt
        # Note: np.dot(v,v) is often faster for this than v[0]**2 + v[1]**2 + v[2]**2
        return np.dot(self._components, self._components)

    def normalize(self) -> 'Vector':
        r"""
        Return a normalized version of the vector (unit vector).
        The resulting vector has the same direction but a magnitude of 1.

        Formula: $\hat{v} = \frac{\vec{v}}{||\vec{v}||}$

        Returns:
        Vector: The normalized vector.

        Raises:
        VectorNormalizationError: If the vector's magnitude is zero (or very close to zero).
        """
        mag = self.magnitude()
        # Use isclose for floating point comparison
        if math.isclose(mag, 0.0, abs_tol=DEFAULT_ATOL):
            raise VectorNormalizationError("Cannot normalize a zero vector.", f"magnitude={mag}")
        # Correctly create a new Vector instance by unpacking the components
        return Vector(*(self._components / mag))
        
    def dot_product(self, other: 'Vector') -> float:
        r"""
        Calculate the dot product (scalar product) of this vector with another.

        Formula: $\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z$

        Parameters:
        other (Vector): The other vector for the dot product.

        Returns:
        float: The dot product.

        Raises:
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
            raise VectorTypeError("Dot product operand must be a Vector.", __class__.__name__, type(other).__name__)
        # Correctly return the result
        return np.dot(self._components, other._components)
        
    def cross_product(self, other: 'Vector') -> 'Vector':
        r"""
        Calculate the cross product of this vector with another.
        The resulting vector is perpendicular to both input vectors.

        Formula: $\vec{a} \times \vec{b} = (a_y b_z - a_z b_y)\hat{i} + (a_z b_x - a_x b_z)\hat{j} + (a_x b_y - a_y b_x)\hat{k}$

        Parameters:
        other (Vector): The other vector for the cross product.

        Returns:
        Vector: The vector resulting from the cross product.

        Raises:
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
            raise VectorTypeError("Cross product operand must be a Vector.", __class__.__name__, type(other).__name__)
        # Calculate using numpy and wrap the result in a new Vector instance
        result_array = np.cross(self._components, other._components)
        return Vector(*result_array)

    def angle_between(self, other: 'Vector', degrees: bool = True) -> float:
        r"""
        Calculate the angle between this vector and another vector.

        Uses the formula: $\theta = \arccos\left(\frac{\vec{a} \cdot \vec{b}}{||\vec{a}|| ||\vec{b}||}\right)$

        Parameters:
        other (Vector): The other vector.
        degrees (bool): If True (default), returns angle in degrees. Otherwise, returns in radians.

        Returns:
        float: The angle in degrees or radians.

        Raises:
        VectorAngleError: If either vector has zero magnitude.
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
            raise VectorTypeError("Angle calculation requires two Vectors.", __class__.__name__, type(other).__name__)
            
        mag_self = self.magnitude()
        mag_other = other.magnitude()
        
        # Use isclose for robust check against zero magnitude
        if math.isclose(mag_self, 0.0, abs_tol=DEFAULT_ATOL) or math.isclose(mag_other, 0.0, abs_tol=DEFAULT_ATOL):
            raise VectorAngleError("Cannot calculate angle with a zero vector.", self, other)
            
        dot_prod = self.dot_product(other)
        # Clamp the cosine value to [-1, 1] to handle potential floating-point errors
        cos_theta = np.clip(dot_prod / (mag_self * mag_other), -1.0, 1.0)
        
        theta_rad = np.arccos(cos_theta)
        
        return np.degrees(theta_rad) if degrees else theta_rad

    # --- Vector Relationship Checks ---
    def is_zero(self, atol: float = DEFAULT_ATOL) -> bool:
        """
        Check if the vector is effectively a zero vector (all components close to zero).

        Parameters:
        atol (float): Absolute tolerance for checking closeness to zero.

        Returns:
        bool: True if the vector is a zero vector within the tolerance, False otherwise.
        """
        return np.allclose(self._components, 0.0, atol=atol)

    def is_parallel(self, other: 'Vector', atol: float = DEFAULT_ATOL) -> bool:
        """
        Check if this vector is parallel (or anti-parallel) to another vector.
        Two vectors are parallel if their cross product is the zero vector.

        Parameters:
        other (Vector): The other vector.
        atol (float): Absolute tolerance for checking if the cross product is close to zero.

        Returns:
        bool: True if vectors are parallel within tolerance, False otherwise.
        
        Raises:
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
             raise VectorTypeError("Parallel check requires two Vectors.", __class__.__name__, type(other).__name__)
        # Check if magnitudes are zero first (zero vector is parallel to everything)
        if self.is_zero(atol) or other.is_zero(atol):
            return True 
        cross_prod = self.cross_product(other)
        return cross_prod.is_zero(atol=atol)

    def is_perpendicular(self, other: 'Vector', atol: float = DEFAULT_ATOL) -> bool:
        """
        Check if this vector is perpendicular (orthogonal) to another vector.
        Two vectors are perpendicular if their dot product is zero.

        Parameters:
        other (Vector): The other vector.
        atol (float): Absolute tolerance for checking if the dot product is close to zero.

        Returns:
        bool: True if vectors are perpendicular within tolerance, False otherwise.
        
        Raises:
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
             raise VectorTypeError("Perpendicular check requires two Vectors.", __class__.__name__, type(other).__name__)
        # Zero vector is considered perpendicular to all vectors
        if self.is_zero(atol) or other.is_zero(atol):
            return True
        dot_prod = self.dot_product(other)
        return math.isclose(dot_prod, 0.0, abs_tol=atol)

    # --- Geometric Operations ---
    def distance_to(self, other: 'Vector') -> float:
        """
        Calculate the Euclidean distance between the points represented by this vector and another vector.

        Parameters:
        other (Vector): The other vector (point).

        Returns:
        float: The distance between the vectors.
        
        Raises:
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
            raise VectorTypeError("Distance calculation requires two Vectors.", __class__.__name__, type(other).__name__)
        return (self - other).magnitude()
        
    def distance_squared_to(self, other: 'Vector') -> float:
        """
        Calculate the squared Euclidean distance between this vector and another.
        More efficient than distance_to if only comparing distances.

        Parameters:
            other (Vector): The other vector (point).

        Returns:
            float: The squared distance between the vectors.

        Raises:
            VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
            raise VectorTypeError("Squared distance calculation requires two Vectors.", __class__.__name__, type(other).__name__)
        return (self - other).magnitude_squared()


    def project_onto(self, other: 'Vector') -> 'Vector':
        r"""
        Calculate the vector projection of this vector onto another vector.

        Formula: $\text{proj}_{\vec{b}} \vec{a} = \frac{\vec{a} \cdot \vec{b}}{||\vec{b}||^2} \vec{b}$

        Parameters:
        other (Vector): The vector onto which to project this vector.

        Returns:
        Vector: The projection vector.

        Raises:
        VectorOperationError: If the vector 'other' is a zero vector.
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
            raise VectorTypeError("Projection requires two Vectors.", __class__.__name__, type(other).__name__)
            
        other_mag_sq = other.magnitude_squared()
        if math.isclose(other_mag_sq, 0.0, abs_tol=DEFAULT_ATOL**2): # Check squared mag against squared tolerance
            raise VectorOperationError("Cannot project onto a zero vector.", self, other)
            
        dot_prod = self.dot_product(other)
        scale_factor = dot_prod / other_mag_sq
        return other * scale_factor # Uses __mul__

    def reject_from(self, other: 'Vector') -> 'Vector':
        """
        Calculate the vector rejection of this vector from another vector.
        This is the component of this vector perpendicular to the 'other' vector.
        Formula: $\text{rej}_{\vec{b}} \vec{a} = \vec{a} - \text{proj}_{\vec{b}} \vec{a}$

        Parameters:
        other (Vector): The vector from which to calculate the rejection.

        Returns:
        Vector: The rejection vector.

        Raises:
        VectorOperationError: If 'other' is a zero vector (via project_onto).
        VectorTypeError: If 'other' is not a Vector instance.
        """
        # Type check happens in project_onto
        return self - self.project_onto(other) # Uses __sub__

    def reflect_across(self, normal: 'Vector') -> 'Vector':
        r"""
        Calculate the reflection of this vector across a plane defined by a normal vector.

        Assumes 'normal' is the normal vector of the reflection plane/line.
        Formula: $\vec{v}_{\text{reflected}} = \vec{v} - 2 \cdot \text{proj}_{\vec{n}} \vec{v}$

        Parameters:
        normal (Vector): The normal vector defining the plane of reflection.

        Returns:
        Vector: The reflected vector.

        Raises:
        VectorOperationError: If 'normal' is a zero vector (via project_onto).
        VectorTypeError: If 'normal' is not a Vector instance.
        """
        # Type check happens in project_onto
        # Note: project_onto expects the vector *onto which* to project.
        # Here, we project self onto the normal.
        return self - 2.0 * self.project_onto(normal) # Uses __sub__ and __mul__

    def lerp(self, other: 'Vector', t: float) -> 'Vector':
        """
        Linearly interpolate between this vector and another vector.
        Result = self * (1 - t) + other * t

        Parameters:
        other (Vector): The target vector to interpolate towards.
        t (float): Interpolation factor. Typically clamped between 0 (returns self) and 1 (returns other).

        Returns:
        Vector: The interpolated vector.
        
        Raises:
        VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
             raise VectorTypeError("Lerp requires two Vectors.", __class__.__name__, type(other).__name__)
        # Clamp t to [0, 1] for standard behavior, though not strictly required mathematically
        # t = max(0.0, min(1.0, t)) # Optional clamping
        return self * (1.0 - t) + other * t # Uses __add__, __mul__

    def rotate(self, angle: float, axis: 'Vector', degrees: bool = True) -> 'Vector':
        r"""
        Rotate this vector around a given axis by a specified angle.

        Uses Rodrigues' rotation formula:
        $\vec{v}_{\text{rot}} = \vec{v} \cos \theta + (\hat{k} \times \vec{v}) \sin \theta + \hat{k} (\hat{k} \cdot \vec{v}) (1 - \cos \theta)$
        where $\theta$ is the angle and $\hat{k}$ is the unit vector along the axis.

        Parameters:
        angle (float): The angle of rotation.
        axis (Vector): The axis (Vector object) around which to rotate.
        degrees (bool): If True (default), 'angle' is interpreted as degrees. Otherwise, radians.

        Returns:
        Vector: The rotated vector.

        Raises:
        VectorOperationError: If 'axis' is a zero vector (cannot normalize).
        VectorTypeError: If 'axis' is not a Vector instance.
        """
        if not isinstance(axis, Vector):
            raise VectorTypeError("Rotation axis must be a Vector.", __class__.__name__, type(axis).__name__)
            
        rad_angle = np.radians(angle) if degrees else float(angle)
        cos_theta = np.cos(rad_angle)
        sin_theta = np.sin(rad_angle)
        
        try:
            unit_axis = axis.normalize()
        except VectorNormalizationError:
            # Reraise with more context
            raise VectorOperationError("Rotation axis cannot be a zero vector.", self, axis) from None

        # Rodrigues' formula implementation
        term1 = self * cos_theta
        term2 = unit_axis.cross_product(self) * sin_theta
        term3 = unit_axis * (unit_axis.dot_product(self) * (1.0 - cos_theta))
        
        return term1 + term2 + term3 # Uses __add__, __mul__

    # --- Conversion Methods ---
    @staticmethod
    def from_tuple(tup: Tuple[float, float, float]) -> 'Vector':
        """Create a Vector from a 3-element tuple."""
        if len(tup) != 3:
             raise VectorInitializationError(f"Tuple must have 3 elements, got {len(tup)}.")
        return Vector(*tup)
    
    @staticmethod
    def from_list(lst: List[float]) -> 'Vector':
        """Create a Vector from a 3-element list."""
        if len(lst) != 3:
            raise VectorInitializationError(f"List must have 3 elements, got {len(lst)}.")
        return Vector(*lst)

    def to_tuple(self) -> Tuple[float, float, float]:
        """Return the vector's components as a tuple (x, y, z)."""
        # Access components directly for slight performance gain over properties
        return (self._components[0], self._components[1], self._components[2])

    def to_list(self) -> List[float]:
        """Return the vector's components as a list [x, y, z]."""
        # Access components directly
        return [self._components[0], self._components[1], self._components[2]]
        
    def to_numpy(self) -> np.ndarray:
        """Return the vector's components as a numpy array."""
        # Return a copy to prevent external modification of internal state
        return self._components.copy()

    # --- Operator Overloading ---
    def __add__(self, other: 'Vector') -> 'Vector':
        """Vector addition (self + other)."""
        if not isinstance(other, Vector):
            # Let Python handle NotImplemented for incompatible types
            return NotImplemented 
        result_array = self._components + other._components
        return Vector(*result_array)

    def __radd__(self, other: Any) -> 'Vector':
        """Reflected vector addition (other + self). Handles cases like sum([v1, v2])."""
        # sum() starts with 0, so handle 0 + Vector case
        if other == 0: 
            return self
        elif isinstance(other, Vector):
            return self.__add__(other) # Delegate to __add__
        return NotImplemented

    def __sub__(self, other: 'Vector') -> 'Vector':
        """Vector subtraction (self - other)."""
        if not isinstance(other, Vector):
            return NotImplemented
        result_array = self._components - other._components
        return Vector(*result_array)
        
    def __rsub__(self, other: 'Vector') -> 'Vector':
        """Reflected vector subtraction (other - self)."""
        if not isinstance(other, Vector):
            return NotImplemented
        # other - self = -(self - other)
        result_array = other._components - self._components
        return Vector(*result_array)


    def __mul__(self, scalar: float) -> 'Vector':
        """Scalar multiplication (vector * scalar)."""
        # Check if scalar is a number (more robust than just float)
        if not isinstance(scalar, (int, float, np.number)): 
             return NotImplemented
        result_array = self._components * float(scalar) # Ensure float mult
        return Vector(*result_array)
    
    def __rmul__(self, scalar: float) -> 'Vector':
        """Reflected scalar multiplication (scalar * vector)."""
        # Delegate to __mul__ as multiplication is commutative
        return self.__mul__(scalar) 
    
    def __truediv__(self, scalar: float) -> 'Vector':
        """Scalar division (vector / scalar)."""
        if not isinstance(scalar, (int, float, np.number)):
             return NotImplemented
        
        fscalar = float(scalar)
        # Check for division by zero AFTER ensuring it's a number
        if math.isclose(fscalar, 0.0, abs_tol=DEFAULT_ATOL):
            raise ZeroDivisionError("Vector division by zero.")
            
        result_array = self._components / fscalar
        return Vector(*result_array)
        
    # --- Comparison Operators (Leveraging @total_ordering) ---
    def __eq__(self, other: object) -> bool:
        """Equality check (self == other). Checks for exact component-wise equality."""
        if not isinstance(other, Vector):
            return NotImplemented # Important for correct inter-type comparisons
        # Use numpy's efficient array comparison
        return np.array_equal(self._components, other._components)
        
    # Note: For floating-point vectors, exact equality '==' can be problematic.
    # Consider adding an is_close() method for tolerance-based comparison.
    def is_close(self, other: 'Vector', rel_tol: float = 1e-09, abs_tol: float = DEFAULT_ATOL) -> bool:
        """
        Check if this vector is close to another vector within specified tolerances.
        Uses component-wise comparison with relative and absolute tolerances.

        Parameters:
            other (Vector): The vector to compare against.
            rel_tol (float): Relative tolerance (see math.isclose documentation).
            abs_tol (float): Absolute tolerance (see math.isclose documentation).

        Returns:
            bool: True if vectors are close, False otherwise.
            
        Raises:
            VectorTypeError: If 'other' is not a Vector instance.
        """
        if not isinstance(other, Vector):
             raise VectorTypeError("is_close requires two Vectors.", __class__.__name__, type(other).__name__)
        return np.allclose(self._components, other._components, rtol=rel_tol, atol=abs_tol)


    def __lt__(self, other: 'Vector') -> bool:
        """Less than comparison (self < other). Compares magnitudes."""
        if not isinstance(other, Vector):
             return NotImplemented
        # Avoid calculating magnitude twice if possible
        # Check if magnitudes are already cached
        mag_self = self._magnitude if self._magnitude is not None else self.magnitude()
        mag_other = other._magnitude if other._magnitude is not None else other.magnitude()
        
        # Handle potential floating point inaccuracies in comparison
        if math.isclose(mag_self, mag_other, abs_tol=DEFAULT_ATOL):
            return False # Consider them equal if very close
        return mag_self < mag_other

    # --- Unary Operators ---
    def __neg__(self) -> 'Vector':
        """Unary negation (-vector)."""
        return Vector(*(-self._components))
    
    def __pos__(self) -> 'Vector':
        """Unary plus (+vector). Returns a copy."""
        return Vector(*self._components) # Or Vector(self)

    def __abs__(self) -> float:
        """Absolute value (abs(vector)). Returns the magnitude."""
        return self.magnitude()
        
    # Note: A previous version returned Vector(abs(x), abs(y), abs(z)).
    # Standard mathematical convention is that abs(vector) or ||vector|| is the magnitude (a scalar).
    # If component-wise abs is needed, a specific method like `component_abs()` could be added.
    def component_abs(self) -> 'Vector':
        """Return a new vector with the absolute values of each component."""
        return Vector(*(np.absolute(self._components)))


    # --- Container Emulation ---
    def __len__(self) -> int:
        """Return the length (number of components), always 3."""
        return 3
        
    def __getitem__(self, index: int) -> float:
        """Get component by index (vector[0] -> x, vector[1] -> y, vector[2] -> z)."""
        try:
            return self._components[index]
        except IndexError:
            # More informative error
            raise IndexError("Vector index out of range (must be 0, 1, or 2).") from None

    def __setitem__(self, index: int, value: float) -> None:
        """Set component by index (vector[0] = value)."""
        try:
            new_val = float(value)
            # Only invalidate if value actually changes
            if self._components[index] != new_val:
                self._components[index] = new_val
                self.invalidate_cache()
        except IndexError:
            raise IndexError("Vector index out of range (must be 0, 1, or 2).") from None
        except (ValueError, TypeError):
             raise TypeError(f"Vector component value must be a number, got {type(value).__name__}.") from None

    def __iter__(self):
        """Return an iterator over the components (x, y, z)."""
        return iter(self._components)

    # --- Boolean Context ---
    def __bool__(self) -> bool:
        """Truthiness test. Returns True if the vector is not the zero vector."""
        # Use the is_zero check for consistency with tolerance
        return not self.is_zero() 

    # --- String Representations ---
    def __repr__(self) -> str:
        """Return the 'official' string representation (Vector(x=..., y=..., z=...))."""
        # Use direct access for speed
        return f"Vector(x={self._components[0]}, y={self._components[1]}, z={self._components[2]})"
    
    def __str__(self) -> str:
        """Return the 'informal' string representation ((x, y, z))."""
        # Use direct access
        return f"({self._components[0]}, {self._components[1]}, {self._components[2]})"

    def __format__(self, format_spec: str) -> str:
        """Support formatted output of vector components (e.g., f'{v:.2f}')."""
        # Apply format_spec to each component
        return f"({self.x:{format_spec}}, {self.y:{format_spec}}, {self.z:{format_spec}})"

    # --- Hashing ---
    # Vectors are mutable (setters, __setitem__), so they should not be hashable.
    __hash__ = None


T = TypeVar('T')

@total_ordering
class DataVector(Sequence[T]):
    """
    A container for holding a fixed-length sequence of data of the same type.
    Supports various operations, including element-wise operations where applicable.
    """

    __slots__ = ('_components', '_dtype')

    def __init__(self, *args: T, dtype: Optional[Type[T]] = None) -> None:
        """
        Initialize a DataVector object.

        Can be initialized in several ways:
        - DataVector(x, y, z, ..., dtype=T): Creates a tuple from the given values.
        - DataVector([x, y, z, ...], dtype=T): Creates a tuple from a list.
        - DataVector((x, y, z, ...), dtype=T): Creates a tuple from a tuple.

        Parameters:
        *args: Arguments for initialization.
        dtype (type, optional): The data type of the components. If None, it is inferred
            from the first argument.  If args is empty, dtype defaults to None.

        Raises:
        VectorInitializationError: If arguments are invalid or incompatible.
        VectorTypeError: If elements in args do not match the specified dtype.
        """

        if not args:
            if dtype is None:
                self._components: Tuple[T, ...] = tuple()
                self._dtype = None
            else:
                self._components = tuple()
                self._dtype = dtype

        if dtype is None:
            self._dtype = type(args[0])
        else:
            self._dtype = dtype

        components: List[T] = []
        for arg in args:
            if not isinstance(arg, self._dtype):
                raise VectorTypeError(
                    f"Expected type {self._dtype}, got {type(arg).__name__}",
                    "DataVector",
                    type(arg).__name__,
                )
            components.append(arg)
        self._components = tuple(components)

    @property
    def dtype(self) -> Optional[Type[T]]:
        """Get the data type of the components."""
        return self._dtype

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._components)

    def __getitem__(self, index: int) -> T:
        """Get the component at the given index."""
        return self._components[index]

    def __setitem__(self, index: int, value: T) -> None:
        """Set the component at the given index."""
        if not isinstance(value, self._dtype):
            raise VectorTypeError(
                f"Expected type {self._dtype}, got {type(value).__name__}",
                "DataVector",
                type(value).__name__,
            )
        temp_list = list(self._components)
        temp_list[index] = value
        self._components = tuple(temp_list)

    def __iter__(self) -> Iterable[T]:
        """Return an iterator over the components."""
        return iter(self._components)

    def __repr__(self) -> str:
        """Return the 'official' string representation."""
        return f"DataVector({', '.join(repr(c) for c in self._components)}, dtype={self._dtype})"

    def __str__(self) -> str:
        """Return the 'informal' string representation."""
        return str(self._components)

    def __format__(self, format_spec: str) -> str:
        """Support formatted output of components."""
        formatted_components = [f"{c:{format_spec}}" for c in self._components]
        return f"({', '.join(formatted_components)})"

    def __eq__(self, other: object) -> bool:
        """Equality check (self == other)."""
        if not isinstance(other, DataVector):
            return NotImplemented
        if self._dtype != other._dtype:
            return False  # Consider different dtypes as not equal
        return self._components == other._components

    def __lt__(self, other: 'DataVector') -> bool:
        """Less than comparison (self < other).  Lexicographical comparison."""
        if not isinstance(other, DataVector):
            return NotImplemented
        if self._dtype != other._dtype:
            return NotImplemented  # Don't compare if types are different
        return self._components < other._components

    def __add__(self, other: 'DataVector') -> 'DataVector':
        """Element-wise addition (or concatenation if applicable)."""
        if not isinstance(other, DataVector):
            return NotImplemented
        if self._dtype != other._dtype:
            raise VectorOperationError("Addition requires DataVectors of the same type.", self, other)
        if len(self) != len(other):
            raise VectorOperationError("Addition requires DataVectors of the same length.", self, other)

        new_components: List[Any] = []
        for i in range(len(self)):
            try:
                new_components.append(self[i] + other[i])
            except TypeError:
                new_components.append(str(self[i]) + str(other[i]))
        return DataVector(*new_components, dtype=self._dtype)

    def __sub__(self, other: 'DataVector') -> 'DataVector':
        """Element-wise subtraction."""
        if not isinstance(other, DataVector):
            return NotImplemented
        if self._dtype != other._dtype:
            raise VectorOperationError("Subtraction requires DataVectors of the same type.", self, other)
        if len(self) != len(other):
            raise VectorOperationError("Subtraction requires DataVectors of the same length.", self, other)

        new_components: List[Any] = []
        for i in range(len(self)):
            try:
                new_components.append(self[i] - other[i])
            except TypeError:
                raise VectorOperationError(
                    f"Subtraction not supported for type {self._dtype}", self, other
                ) from None
        return DataVector(*new_components, dtype=self._dtype)

    def __mul__(self, scalar: Any) -> 'DataVector':
        """Scalar multiplication (or repetition if applicable)."""
        if not isinstance(scalar, (int, float, np.number)):
            return NotImplemented

        new_components: List[Any] = []
        for component in self._components:
            try:
                new_components.append(component * scalar)
            except TypeError:
                new_components.append(str(component) * scalar)
        return DataVector(*new_components, dtype=self._dtype)

    def __rmul__(self, scalar: Any) -> 'DataVector':
        """Reflected scalar multiplication."""
        return self * scalar

    def __truediv__(self, scalar: Any) -> 'DataVector':
        """Scalar division."""
        if not isinstance(scalar, (int, float, np.number)):
            return NotImplemented
        try:
            scalar = float(scalar)  # Ensure float for division
            new_components: List[Any] = [c / scalar for c in self._components]
            return DataVector(*new_components, dtype=self._dtype)
        except ZeroDivisionError:
            raise ZeroDivisionError("Division by zero") from None
        except TypeError:
            raise VectorOperationError(
                f"Division not supported for type {self._dtype}", self, scalar
            ) from None

    def sum(self) -> Any:
        """
        Calculate the sum of the components.

        Returns:
            Any: The sum of the components.

        Raises:
            TypeError: If the components do not support addition.
        """
        if not self._components:
            return 0  # Or raise an error, depending on desired behavior for empty tuple
        try:
            return sum(self._components)
        except TypeError:
            raise VectorOperationError(f"Summation not supported for type {self._dtype}", self) from None

    def mean(self) -> Any:
        """
        Calculate the mean (average) of the components.

        Returns:
            Any: The mean of the components.

        Raises:
            TypeError: If the components do not support addition and division.
        """
        if not self._components:
            return 0  # Or raise an error
        try:
            return sum(self._components) / len(self._components)
        except TypeError:
            raise VectorOperationError(f"Mean calculation not supported for type {self._dtype}", self) from None