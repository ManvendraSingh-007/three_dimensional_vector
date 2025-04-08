import math

class Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5
    
    def cross_product(self, other: 'Vector') -> 'Vector':
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)
    
    def dot_product(self, other: 'Vector') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def normalize(self) -> 'Vector':
        magnitude = self.magnitude()
        return Vector(self.x / magnitude, self.y / magnitude, self.z / magnitude)
    
    def angle_between(self, other: 'Vector', degrees: bool = True) -> float:
        cos_theta = self.dot_product(other) / (self.magnitude() * other.magnitude())
        cos_theta = min(max(cos_theta, -1), 1)
        theta = math.acos(cos_theta)
        return math.degrees(theta) if degrees else theta
    
    def is_zero(self) -> bool:
        return self.x == 0 and self.y == 0 and self.z == 0
    
    def is_parallel(self, other: 'Vector') -> bool:
        return self.cross_product(other).is_zero()
    
    def is_perpendicular(self, other: 'Vector') -> bool:
        return self.dot_product(other) == 0
    
    def is_orthogonal(self, other: 'Vector') -> bool:
        return self.cross_product(other).magnitude() == 0
    
    def distance_to(self, other: 'Vector') -> float:
        return (self - other).magnitude()
    
    def project_onto(self, other: 'Vector') -> 'Vector':
        projection_magnitude = self.dot_product(other) / other.magnitude()
        return other * projection_magnitude
    
    def reject_from(self, other: 'Vector') -> 'Vector':
        return self - self.project_onto(other)
    
    def reflect_across(self, normal: 'Vector') -> 'Vector':
        return self - 2 * self.project_onto(normal)
    
    def from_tuple(self, tup: tuple) -> 'Vector':
        return Vector(*tup)
       
    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)
    
    def from_list(self, lst: list) -> 'Vector':
        return Vector(*lst)
    
    def to_list(self) -> list:
        return [self.x, self.y, self.z]
    
    def lerp(self, other: 'Vector', t: float) -> 'Vector':
        return self * (1 - t) + other * t
    
    def rotate(self, angle: float, axis: 'Vector') -> 'Vector':
        rad_angle = math.radians(angle)
        cos_theta = math.cos(rad_angle)
        sin_theta = math.sin(rad_angle)
        unit_axis = axis.normalize()

        return self.__mul__(cos_theta) + (unit_axis.cross_product(self)).__mul__(sin_theta) + unit_axis.__mul__((self.dot_product(unit_axis) * (1 - cos_theta)))
    def __add__(self, *other: 'Vector') -> 'Vector':
        x = self.x
        y = self.y
        z = self.z
        for vector in other:
            x += vector.x
            y += vector.y
            z += vector.z
        return Vector(x, y, z)
    
    def __sub__(self, *other: 'Vector') -> 'Vector':
        x = self.x
        y = self.y
        z = self.z
        for vector in other:
            x -= vector.x
            y -= vector.y
            z -= vector.z
        return Vector(x, y, z)
    
    def __mul__(self, scalar: float) -> 'Vector':
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector':
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector':
        return self.__mul__(1 / scalar)
    
    def __eq__(self, other: 'Vector') -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __neg__(self) -> 'Vector':
        return self.__mul__(-1)
    
    def __abs__(self) -> 'Vector':
        return Vector(abs(self.x), abs(self.y), abs(self.z))
    
    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"
    
    def __format__(self, format_spec: str) -> str:
        return f"Vector({self.x:.{format_spec}f}, {self.y:.{format_spec}f}, {self.z:.{format_spec}f})"
    
    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]
       
    def __setitem__(self, index: int, value: float) -> None:
        (self.x, self.y, self.z)[index] = value

    def __len__(self) -> int:
        return 3
    
    def __bool__(self) -> bool:
        return self.x!= 0 or self.y!= 0 or self.z!= 0
    
    def __iter__(self):
        return iter((self.x, self.y, self.z))
    
    def __ne__(self, other: 'Vector') -> bool:
        return not self.__eq__(other)
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"
    
    __hash__ = None