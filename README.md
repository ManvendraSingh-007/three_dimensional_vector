# Python 3D Vector Library ✨

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Status](https://img.shields.io/badge/status-active-yellow)
[![NumPy](https://img.shields.io/badge/dependency-NumPy-blueviolet)](https://numpy.org/)


## Description

A Python library implementing a 3-dimensional `Vector` class with comprehensive methods for common vector operations and calculations. It leverages NumPy for performance and provides a clean, Pythonic interface.

## Features

* **Initialization:** Flexible constructor (coordinates, sequences, other vectors). Includes class methods for common vectors (`zero`, `ones`, `unit_x`, `unit_y`, `unit_z`).
* **Core Operations:** Magnitude (cached), normalization, dot product, cross product.
* **Arithmetic:** Operator overloading for addition, subtraction, scalar multiplication/division (`+`, `-`, `*`, `/`).
* **Geometric Operations:** Angle between vectors, projection, rejection, reflection, rotation, linear interpolation (lerp).
* **Comparisons:** Equality (`==`), magnitude-based ordering (`<`, `<=`, `>`, `>=`), closeness check with tolerance (`is_close`). Methods for checking parallelism and perpendicularity (`is_parallel`, `is_perpendicular`).
* **Utility Methods:** Distance calculation, conversion to/from tuples, lists, NumPy arrays (`to_tuple`, `to_list`, `to_numpy`, `from_tuple`, `from_list`). Component access via indexing (`[]`).
* **Pythonic Interface:** Supports `len()`, `abs()` (magnitude), boolean checks (`bool()`), iteration, string representation (`str`, `repr`), and formatting (`format`).

## Installation

**Dependency:** This library requires [NumPy](https://numpy.org/). Install it if you haven't already:
```bash
pip install numpy
```
Then, simply copy the vector.py and vector_errors.py files into your project directory.
```python
from vector import Vector
```

## Initilisation
```python
from vector import Vector
import numpy as np

# From coordinates
v1 = Vector(1.0, 2.0, 3.0)
v_xy = Vector(5.0, -2.0) # z defaults to 0.0
v_x = Vector(10.0)       # y and z default to 0.0

# From sequences
v_list = Vector([4, 5, 6])
v_tuple = Vector((7, 8, 9))
v_np = Vector(np.array([1.1, 2.2, 3.3]))

# Common vectors
zero_vec = Vector.zero()     # Vector(0, 0, 0)
ones_vec = Vector.ones()     # Vector(1, 1, 1)
unit_x = Vector.unit_x()   # Vector(1, 0, 0)

# Copy constructor
v_copy = Vector(v1)

print(v1)
print(v_list)
print(unit_x)
```

## Basic Operations
```python
a = Vector(1, 2, 3)
b = Vector(4, 5, 6)

# Arithmetic
print(f"Sum: {a + b}")
print(f"Difference: {a - b}")
print(f"Scaled: {a * 3}")
print(f"Divided: {b / 2}")

# Dot and Cross Product
dot_p = a.dot_product(b)
cross_p = a.cross_product(b)
print(f"Dot Product: {dot_p}")
print(f"Cross Product: {cross_p}") # Note: Returns a Vector object

# Magnitude & Normalization
mag_a = a.magnitude()
mag_sq_a = a.magnitude_squared() # More efficient for comparisons
norm_a = a.normalize()
print(f"Magnitude of a: {mag_a:.3f}")
print(f"Normalized a: {norm_a:.3f}") # Uses __format__
print(f"Normalized Magnitude: {norm_a.magnitude()}") # Should be ~1.0
```

## Geometric Calculations & Comparisons
```python
v_i = Vector(1, 0, 0)
v_j = Vector(0, 1, 0)
v_k = Vector(0, 0, 1)
v_diag = Vector(1, 1, 0)

# Angle
angle = v_i.angle_between(v_diag)
print(f"Angle between (1,0,0) and (1,1,0): {angle:.2f} degrees")

# Projection, Rejection, Reflection
proj = v_diag.project_onto(v_i)
rej = v_diag.reject_from(v_i)
refl = v_diag.reflect_across(v_j) # Reflect across XZ plane (normal is Y axis)
print(f"Projection of {v_diag} onto {v_i}: {proj}")
print(f"Rejection of {v_diag} from {v_i}: {rej}")
print(f"Reflection of {v_diag} across Y-normal: {refl}")

# Rotation
rotated = v_j.rotate(90, axis=v_k) # Rotate (0,1,0) 90 deg around Z axis
print(f"Rotation of {v_j} around Z by 90 deg: {rotated}") # Should be (-1, 0, 0)

# Comparisons (use is_close for floats!)
v_almost_j = Vector(1e-9, 1.0, 0)
print(f"v_j == v_almost_j: {v_j == v_almost_j}") # False (exact comparison)
print(f"v_j.is_close(v_almost_j): {v_j.is_close(v_almost_j)}") # True (within tolerance)

# Parallel / Perpendicular (uses tolerance)
print(f"v_i parallel to Vector(5,0,0)? {v_i.is_parallel(Vector(5,0,0))}") # True
print(f"v_i perpendicular to v_j? {v_i.is_perpendicular(v_j)}") # True
```

## Other Utilities
```py
p1 = Vector(1, 1, 1)
p2 = Vector(4, 5, 1)

# Distance
dist = p1.distance_to(p2)
print(f"Distance between {p1} and {p2}: {dist}") # Should be 5.0

# Lerp
midpoint = p1.lerp(p2, 0.5)
print(f"Midpoint (lerp t=0.5): {midpoint}") # Should be (2.5, 3.0, 1.0)

# Conversion
my_tuple = p1.to_tuple()
my_list = p1.to_list()
my_array = p1.to_numpy()
print(f"Tuple: {my_tuple}, List: {my_list}, NumPy: {my_array}")

# Indexing and Iteration
print(f"p1[0] (x): {p1[0]}")
print("Components:")
for component in p1:
    print(f"- {component}")
```

## API Overview

*(This provides a brief summary. See the source code docstrings for full details.)*

* **`Vector(*args)`**: Constructor (see Initialization examples).
* **`.x`, `.y`, `.z`**: Properties to get/set components.
* **`magnitude()`**: Returns the vector length (`float`).
* **`magnitude_squared()`**: Returns squared length (`float`, faster for comparisons).
* **`normalize()`**: Returns a unit vector (`Vector`) with the same direction.
* **`dot_product(other)`**: Returns the scalar dot product (`float`).
* **`cross_product(other)`**: Returns the vector cross product (`Vector`).
* **`angle_between(other, degrees=True)`**: Returns the angle to `other` (`float`).
* **`is_zero(atol=...)`**: Checks if vector is close to zero vector (`bool`).
* **`is_parallel(other, atol=...)`**: Checks if parallel/anti-parallel to `other` (`bool`).
* **`is_perpendicular(other, atol=...)`**: Checks if orthogonal to `other` (`bool`).
* **`distance_to(other)`**: Returns Euclidean distance (`float`).
* **`distance_squared_to(other)`**: Returns squared distance (`float`).
* **`project_onto(other)`**: Returns projection of self onto `other` (`Vector`).
* **`reject_from(other)`**: Returns rejection of self from `other` (`Vector`).
* **`reflect_across(normal)`**: Returns reflection across plane with `normal` (`Vector`).
* **`lerp(other, t)`**: Linear interpolation towards `other` (`Vector`).
* **`rotate(angle, axis, degrees=True)`**: Rotates vector around `axis` (`Vector`).
* **`is_close(other, rel_tol=..., abs_tol=...)`**: Checks component-wise closeness (`bool`).
* **`component_abs()`**: Returns new `Vector` with absolute value of each component.
* **`to_tuple()`, `to_list()`, `to_numpy()`**: Conversion methods.
* **`from_tuple(tup)`, `from_list(lst)`**: Static methods for creation.
* **Special methods**: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__eq__`, `__lt__` (and others via `@total_ordering`), `__neg__`, `__pos__`, `__abs__` (magnitude), `__len__`, `__getitem__`, `__setitem__`, `__iter__`, `__bool__`, `__str__`, `__repr__`, `__format__`.

## Contributing
Contributions are welcome! Please feel free to report bugs, suggest enhancements, or submit pull requests for new features, optimizations, tests, or documentation improvements.
See Contributing Guide for more details on the development process.

## License
MIT License. See [LICENSE](LICENSE) for details.
