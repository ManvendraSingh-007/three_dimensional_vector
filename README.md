# Python 3D Vector Library ✨

## Description

This is a Python library implementing a 3D vector class (`Vector`) with various useful methods for vector operations. It provides a convenient way to perform common vector calculations in 3-dimensional space.

## Features ⚡

The `Vector` class includes the following functionalities:

* **Initialization:** Creating a vector with x, y, and z components.
* **Basic Operations:**
    * Addition (`+`)
    * Subtraction (`-`)
    * Scalar Multiplication (`*`)
    * Scalar Division (`/`)
    * Negation (`-`)
    * Absolute Value (`abs`)
* **Magnitude:** Calculating the length of the vector.
* **Dot Product:** Calculating the dot product of two vectors.
* **Cross Product:** Calculating the cross product of two vectors.
* **Normalization:** Obtaining a unit vector in the same direction.
* **Angle Between:** Calculating the angle between two vectors (in degrees or radians).
* **Zero Vector Check**: Check if the vector is a zero vector.
* **Parallelism and Perpendicularity:** Checking if two vectors are parallel or perpendicular.
* **Distance:** Calculating the distance between two vectors.
* **Projection and Rejection:** Projecting and rejecting a vector onto another vector.
* **Reflection:** Reflecting a vector across a normal vector.
* **Linear Interpolation (Lerp):** Calculates a vector between two vectors.
* **Rotation:** Rotating a vector by a specified angle around an axis.
* **Conversions:**
    * Converting to and from tuples.
    * Converting to and from lists.
* **Equality Check:** Checking if two vectors are equal (`==`).
* **Indexing:** Accessing vector components using indices (e.g., `v[0]`).
* **Iteration:** Iterating through the components of a vector.
* **String Representation:** User-friendly string representation of a vector.
* **Custom Formatting:** Use f-string formatting to control how vectors are printed.

## Usage 🚀

### Installation

Since this is a single-file library, you can simply copy the `vector.py` file into your project directory. No formal installation is required.

### Example

```python
import math
from vector import Vector  # Assuming you've saved the code in vector.py

# Create vectors
v1 = Vector(1.0, 2.0, 3.0)
v2 = Vector(4.0, 5.0, 6.0)

# Basic operations
v3 = v1 + v2
v4 = v2 - v1
v5 = v1 * 2.0
v6 = v2 / 2.0

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v3 (v1 + v2): {v3}")
print(f"v4 (v2 - v1): {v4}")
print(f"v5 (v1 * 2): {v5}")
print(f"v6 (v2 / 2): {v6}")

# Magnitude
mag_v1 = v1.magnitude()
print(f"Magnitude of v1: {mag_v1:.2f}")

# Dot product
dot_product = v1.dot_product(v2)
print(f"Dot product of v1 and v2: {dot_product:.2f}")

# Cross product
cross_product = v1.cross_product(v2)
print(f"Cross product of v1 and v2: {cross_product}")

# Normalize
v1_normalized = v1.normalize()
print(f"Normalized v1: {v1_normalized:.2f}")

# Angle between
angle_degrees = v1.angle_between(v2)
angle_radians = v1.angle_between(v2, degrees=False)
print(f"Angle between v1 and v2 (degrees): {angle_degrees:.2f}")
print(f"Angle between v1 and v2 (radians): {angle_radians:.2f}")

# Check if zero vector
v_zero = Vector(0, 0, 0)
print(f"Is v1 zero vector: {v1.is_zero()}")
print(f"Is v_zero zero vector: {v_zero.is_zero()}")

# Check if vectors are parallel
v3 = v1 * 2
print(f"Are v1 and v3 parallel? {v1.is_parallel(v3)}")
print(f"Are v1 and v2 parallel? {v1.is_parallel(v2)}")

# Check if vectors are perpendicular
print(f"Are v1 and v2 perpendicular? {v1.is_perpendicular(v2)}")

# Distance
distance = v1.distance_to(v2)
print(f"Distance between v1 and v2: {distance:.2f}")

# Projection and rejection
v_proj = v1.project_onto(v2)
v_rej = v1.reject_from(v2)
print(f"Projection of v1 onto v2: {v_proj:.2f}")
print(f"Rejection of v1 from v2: {v_rej:.2f}")

# Reflection
normal_vector = Vector(0.0, 1.0, 0.0)  # Reflect across the Y-axis
v1_reflected = v1.reflect_across(normal_vector)
print(f"Reflection of v1 across Y-axis: {v1_reflected}")

# Linear interpolation (Lerp)
v_lerp = v1.lerp(v2, 0.5)
print(f"Lerp between v1 and v2 at t=0.5: {v_lerp}")

# Rotation
axis = Vector(0, 1, 0)  # Rotate around the Y-axis
rotated_v1 = v1.rotate(90, axis)
print(f"Rotated v1 by 90 degrees around Y-axis: {rotated_v1:.2f}")

# Conversion to/from tuple/list
v1_tuple = v1.to_tuple()
v2_list = v2.to_list()
v3_from_tuple = Vector.from_tuple((7, 8, 9))
v4_from_list = Vector.from_list([10, 11, 12])
print(f"v1 as tuple: {v1_tuple}")
print(f"v2 as list: {v2_list}")
print(f"v3 from tuple: {v3_from_tuple}")
print(f"v4 from list: {v4_from_list}")

# Equality check
v7 = Vector(1, 2, 3)
print(f"Is v1 equal to v7? {v1 == v7}")
print(f"Is v1 equal to v2? {v1 == v2}")

# Indexing
print(f"v1[0]: {v1[0]:.2f}")
print(f"v1[1]: {v1[1]:.2f}")
print(f"v1[2]: {v1[2]:.2f}")

# Iteration
print("Components of v1:")
for component in v1:
    print(f"{component:.2f}")