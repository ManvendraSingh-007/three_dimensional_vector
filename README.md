# Python 3D Vector Library ✨

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Status](https://img.shields.io/badge/status-active%20development-yellow)

## Description
This is a Python library implementing a 3D vector class (Vector) with various useful methods for vector operations. It provides a convenient way to perform common vector calculations in 3-dimensional space.

## Project Status 🚧

This project is currently in active development. While the core functionality is stable and well-tested, we're working on:

- Additional optimization opportunities
- Expanded documentation and examples
- More advanced vector operations
- Better integration with scientific Python ecosystems

**We welcome contributions!** See the [Contributing](https://github.com/ManvendraSingh-007/vector/edit/master/README.md#contributing) section below.

## Features

- 🧮 Basic vector operations (addition, subtraction, multiplication, division)
- 📏 Geometric calculations (magnitude, normalization, dot/cross products)
- 📐 Advanced operations (projection, rejection, reflection, rotation)
- 🔄 Conversion methods (to/from tuples, lists)
- 🔍 Comparison and ordering based on magnitude
- ⚡ Magnitude caching for performance
- 🐍 Pythonic interface with special methods

## Installation

Simply copy the `vector.py` file into your project and import it:

```python
from vector import Vector
```

## Different ways to create vectors
```python
v1 = Vector(1, 2, 3)                 # Using coordinates
v2 = Vector.zero()                   # Zero vector (0, 0, 0)
v3 = Vector.unit()                   # Unit vector (1, 1, 1)
v4 = Vector.from_tuple((4, 5, 6))    # From tuple
```

## Vector operations
```python
a = Vector(1, 2, 3)
b = Vector(4, 5, 6)

# Basic arithmetic
sum_vec = a + b
diff_vec = a - b
scaled_vec = a * 2
divided_vec = a / 2

# Dot and cross products
dot_product = a.dot_product(b)
cross_product = a.cross_product(b)

# Magnitude and normalization
magnitude = a.magnitude()
normalized = a.normalize()
```

## Geometric calculation
```python
# Angle between vectors
angle_deg = a.angle_between(b)  # in degrees
angle_rad = a.angle_between(b, degrees=False)  # in radians

# Projections
projection = a.project_onto(b)
rejection = a.reject_from(b)

# Reflection
normal = Vector(0, 1, 0)
reflected = a.reflect_across(normal)

# Rotation
axis = Vector(1, 0, 0)
rotated = a.rotate(45, axis)  # 45 degree rotation around x-axis
```

## Methods Overview
`magnitude()`

`normalize()`

`dot_product(other)`

`cross_product(other)`

`angle_between(other, degrees=True)`

`project_onto(other)`

`reject_from(other)`

`reflect_across(normal)`

`rotate(angle, axis)`

`is_zero()`, `is_parallel(other)`, `is_perpendicular(other)`, `is_orthogonal(other)`

`distance_to(other)`

`lerp(other, t)`

## Operator Overloading
`+`, `-` for vector addition/subtraction

`*`, `/` for scalar multiplication/division

`==`, `<` for comparisons

`[]` for indexing

`len()` for number of dimensions

`bool()` for zero-check

`str()` and `repr()` for printing

`format()` for custom decimal formatting

## Contributing

We'd love your help to make this vector class even better! Here's how you can contribute:

1. **Report bugs** - Open an issue if you find any problems
2. **Suggest enhancements** - Have an idea for improvement? Let us know!
3. **Submit pull requests** - We welcome code contributions for:
   - New vector operations
   - Performance optimizations
   - Better documentation
   - Additional test cases

### Getting Started with Development

1. Fork the repository
2. Create a new branch for your feature/fix
3. Write your code (with tests!)
4. Submit a pull request

Please ensure your code follows the existing style and includes appropriate tests.

## License

MIT License. See [LICENSE](LICENSE) for details.
