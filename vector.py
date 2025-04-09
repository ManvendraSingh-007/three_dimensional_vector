import math
from functools import total_ordering

@total_ordering
class Vector:
    
    def __init__(self, x: float, y: float, z: float) -> None:
        """
        Initialize a Vector object with given x, y, and z coordinates.

        Parameters:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
        z (float): The z-coordinate of the vector.

        Returns:
        None: The function does not return any value. It initializes the Vector object.
        """
        self.__x = x
        self.__y = y
        self.__z = z
        self._magnitude = None

    @property
    def x(self) -> float:
        """
        Get the x-coordinate of the vector.

        Returns:
        float: The x-coordinate of the vector.
        """
        return self.__x
    
    @x.setter
    def x(self, x: float) -> None:
        """
        Set the x-coordinate of the vector and update its magnitude.

        This setter method assigns a new value to the x-coordinate of the vector
        and recalculates the vector's magnitude.

        Parameters:
        x (float): The new x-coordinate of the vector.

        Returns:
        None
        """
        if self.__x != x: # Only update magnitude if the x-coordinate has changed
            self.__x = x
            self.invalidate_cache() # Invalidate the cache for the magnitude

    @property
    def y(self) -> float:
        """
        Get the y-coordinate of the vector.

        Returns:
        float: The y-coordinate of the vector.
        """

        return self.__y
    
    @y.setter
    def y(self, y: float) -> None:
        """
        Set the y-coordinate of the vector and update its magnitude.

        This setter method assigns a new value to the y-coordinate of the vector
        and recalculates the vector's magnitude.

        Parameters:
        y (float): The new y-coordinate of the vector.

        Returns:
        None
        """
        if self.__y != y: # Only update magnitude if the y-coordinate has changed
            self.__y = y
            self.invalidate_cache() # Invalidate the cache for the magnitude

    @property
    def z(self) -> float:
        """
        Get the z-coordinate of the vector.

        Returns:
        float: The z-coordinate of the vector.
        """
        return self.__z
    
    @z.setter
    def z(self, z: float) -> None:
        """
        Set the z-coordinate of the vector and update its magnitude.

        This setter method assigns a new value to the z-coordinate of the vector
        and recalculates the vector's magnitude.

        Parameters:
        z (float): The new z-coordinate of the vector.

        Returns:
        None
        """
        
        if self.__z != z: # Only update magnitude if the z-coordinate has changed
            self.__z = z
            self.invalidate_cache() # Invalidate the cache for the magnitude

    def invalidate_cache(self) -> None:
        self._magnitude = None # Invalidate the cache for the magnitude

    @classmethod
    def zero(cls) -> 'Vector':
        """
        Create and return a zero vector.

        This class method returns a new Vector object with all its components (x, y, z) set to zero.
        It's a convenient way to create a vector representing the origin point in 3D space.

        Returns:
        Vector: A new Vector object with x, y, and z components all set to zero.
        """
        return cls(0, 0, 0) # Return a zero vector

    @classmethod
    def unit(cls) -> 'Vector':
        """
        Create and return a unit vector.

        This class method returns a new Vector object with all its components (x, y, z) set to one.
        It's a convenient way to create a vector representing a direction in 3D space.

        Returns:
        Vector: A new Vector object with x, y, and z components all set to one.
        """
        return cls(1, 1, 1) # Return a unit vector
    
    @classmethod
    def from_coordinates(cls, x: float, y: float, z: float) -> 'Vector':
        """
        Create and return a new Vector object from given x, y, and z coordinates.

        Parameters:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
        z (float): The z-coordinate of the vector.

        Returns:
        Vector: A new Vector object with the given coordinates.
        """     
        return cls(x, y, z) # Return a new vector with given coordinates
    
    @classmethod
    def from_tuple(cls, coordinates: tuple) -> 'Vector':
        """
        Create a new Vector object from a tuple of coordinates.

        This class method takes a tuple containing three float values (x, y, z) and creates a new Vector object with those coordinates.

        Parameters:
        coordinates (tuple): A tuple containing three float values representing the x, y, and z coordinates of the new Vector object.

        Returns:
        Vector: A new Vector object with the provided coordinates.

        Raises:
        ValueError: If the tuple does not contain exactly three elements.
        """

        if len(coordinates) == 3:
            return cls(*coordinates) # Return a new vector from given tuple of coordinates
        else:
            raise ValueError("Coordinates must be a tuple of length 3")

    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.

        The magnitude of a vector is the square root of the sum of the squares of its components.
        In this case, the magnitude is calculated using the formula: sqrt(x^2 + y^2 + z^2).

        Parameters:
        self (Vector): The vector object for which the magnitude is to be calculated.

        Returns:
        float: The magnitude of the vector.
        """
        
        if self._magnitude == None: # Calculate magnitude if not already calculated
            self._magnitude = (self.x ** 2 + self.y ** 2 + self.z ** 2) ** 0.5

        return self._magnitude # Return magnitude
        

    
    def cross_product(self, other: 'Vector') -> 'Vector':
        """
        Calculate the cross product of the current vector with another vector.

        The cross product of two vectors results in a new vector that is perpendicular to both original vectors.
        The magnitude of the resulting vector is equal to the product of the magnitudes of the original vectors and the sine of the angle between them.

        Parameters:
        other (Vector): The other vector with which to calculate the cross product.

        Returns:
        Vector: A new vector that is the result of the cross product operation.
        """
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return Vector(x, y, z)

    
    def dot_product(self, other: 'Vector') -> float:
        """
        Calculate the dot product of the current vector with another vector.

        The dot product of two vectors is a scalar value that represents the product of their magnitudes and the cosine of the angle between them.
        It is calculated using the formula: x1*x2 + y1*y2 + z1*z2.

        Parameters:
        other (Vector): The other vector with which to calculate the dot product.

        Returns:
        float: The dot product of the current vector and the other vector.
        """
        return self.x * other.x + self.y * other.y + self.z * other.z

    
    def normalize(self) -> 'Vector':
        """
        Normalize the current vector to a unit vector (magnitude of 1).

        The normalization process involves dividing each component of the vector by its magnitude.
        The resulting vector will have the same direction as the original vector but a magnitude of 1.

        Parameters:
        self (Vector): The vector object to be normalized.

        Returns:
        Vector: A new vector that is the normalized version of the original vector.
        """
        magnitude = self.magnitude()
        try:
            return Vector(self.x / magnitude, self.y / magnitude, self.z / magnitude)
        except ZeroDivisionError:
            raise ValueError("Cannot normalize a zero vector")

    
    def angle_between(self, other: 'Vector', degrees: bool = True) -> float:
        """
        Calculate the angle between the current vector and another vector.

        This function calculates the angle between two vectors in 3D space. It uses the dot product and magnitude of the vectors to determine the angle.
        If the `degrees` parameter is set to `True`, the function returns the angle in degrees. Otherwise, it returns the angle in radians.

        Parameters:
        other (Vector): The other vector with which to calculate the angle.
        degrees (bool): A flag indicating whether to return the angle in degrees (default is True).

        Returns:
        float: The angle between the current vector and the other vector, in degrees or radians depending on the `degrees` parameter.
        """
        if (self.magnitude() and other.magnitude()) != 0:
            cos_theta = self.dot_product(other) / (self.magnitude() * other.magnitude())
            cos_theta = min(max(cos_theta, -1), 1)
            theta = math.acos(cos_theta)
            return math.degrees(theta) if degrees else theta
        else:
            raise ValueError("Cannot calculate angle between zero vectors")

    
    def is_zero(self) -> bool:
        """
        Check if the current vector is a zero vector.

        A zero vector is a vector with all its components equal to zero.
        This function checks if the current vector's x, y, and z components are all zero.

        Parameters:
        self (Vector): The vector object to be checked.

        Returns:
        bool: True if the vector is a zero vector, False otherwise.
        """
        return self.x == 0 and self.y == 0 and self.z == 0

    
    def is_parallel(self, other: 'Vector') -> bool:
        """
        Check if the current vector is parallel to another vector.

        This function calculates the cross product of the current vector and the other vector.
        If the cross product is a zero vector, it means the vectors are either parallel or anti-parallel.
        In this case, the function returns True. Otherwise, it returns False.

        Parameters:
        other (Vector): The other vector to compare with the current vector.

        Returns:
        bool: True if the current vector is parallel to the other vector, False otherwise.
        """
        return self.cross_product(other).is_zero()

    
    def is_perpendicular(self, other: 'Vector') -> bool:
        """
        Check if the current vector is perpendicular to another vector.

        This function calculates the dot product of the current vector and the other vector.
        If the dot product is zero, it means the vectors are perpendicular to each other.
        In this case, the function returns True. Otherwise, it returns False.

        Parameters:
        other (Vector): The other vector to compare with the current vector.

        Returns:
        bool: True if the current vector is perpendicular to the other vector, False otherwise.
        """
        return self.dot_product(other) == 0

    
    def is_orthogonal(self, other: 'Vector') -> bool:
        """
        Check if the current vector is orthogonal to another vector.

        This function calculates the cross product of the current vector and the other vector.
        If the magnitude of the cross product is zero, it means the vectors are orthogonal to each other.
        In this case, the function returns True. Otherwise, it returns False.

        Parameters:
        other (Vector): The other vector to compare with the current vector.

        Returns:
        bool: True if the current vector is orthogonal to the other vector, False otherwise.
        """
        return self.cross_product(other).magnitude() == 0

    
    def distance_to(self, other: 'Vector') -> float:
        """
        Calculate the Euclidean distance between the current vector and another vector.

        The Euclidean distance between two points in 3D space is the straight-line distance between them.
        This function calculates the distance between the current vector and another vector using the formula: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2).

        Parameters:
        other (Vector): The other vector to calculate the distance to.

        Returns:
        float: The Euclidean distance between the current vector and the other vector.
        """
        return (self - other).magnitude()

    
    def project_onto(self, other: 'Vector') -> 'Vector':
        """
        Project the current vector onto another vector.

        This function calculates the projection of the current vector onto another vector.
        The projection is a vector that has the same direction as the other vector and a magnitude
        that represents the component of the current vector along the direction of the other vector.

        Parameters:
        other (Vector): The vector onto which to project the current vector.

        Returns:
        Vector: A new vector that represents the projection of the current vector onto the other vector.
        """
        projection_magnitude = self.dot_product(other) / other.magnitude()
        return other * projection_magnitude

    
    def reject_from(self, other: 'Vector') -> 'Vector':
        """
        Calculate the rejection of the current vector from another vector.

        The rejection of a vector from another vector is a vector that has the same direction as the other vector
        but with a magnitude that represents the component of the current vector that is perpendicular to the other vector.
        This function calculates the rejection by subtracting the projection of the current vector onto the other vector from the current vector.

        Parameters:
        other (Vector): The vector from which to calculate the rejection.

        Returns:
        Vector: A new vector that represents the rejection of the current vector from the other vector.
        """
        return self - self.project_onto(other)

    
    def reflect_across(self, normal: 'Vector') -> 'Vector':
        """
        Calculate the reflection of the current vector across a given normal vector.

        This function calculates the reflection of the current vector across a given normal vector.
        The reflection is a vector that has the same magnitude as the original vector but is directed
        in the opposite direction to the reflected ray.

        Parameters:
        normal (Vector): The normal vector across which to reflect the current vector.

        Returns:
        Vector: A new vector that represents the reflection of the current vector across the given normal vector.
        """
        return self - 2 * self.project_onto(normal)

    
    def from_tuple(self, tup: tuple) -> 'Vector':
        """
        Create a new Vector object from a tuple.

        This function takes a tuple of three float values (x, y, z) and creates a new Vector object with the given coordinates.

        Parameters:
        tup (tuple): A tuple containing three float values representing the x, y, and z coordinates of the new Vector object.

        Returns:
        Vector: A new Vector object with the given coordinates.
        """
        return Vector(*tup)

       
    def to_tuple(self) -> tuple:
        """
        Convert the Vector object to a tuple.

        This method returns a tuple containing the x, y, and z coordinates of the Vector object.

        Parameters:
        self (Vector): The Vector object to be converted.

        Returns:
        tuple: A tuple containing the x, y, and z coordinates of the Vector object.
        """
        return (self.x, self.y, self.z)

    
    def from_list(self, lst: list) -> 'Vector':
        """
        Create a new Vector object from a list.

        This function takes a list of three float values (x, y, z) and creates a new Vector object with the given coordinates.

        Parameters:
        lst (list): A list containing three float values representing the x, y, and z coordinates of the new Vector object.

        Returns:
        Vector: A new Vector object with the given coordinates.
        """
        return Vector(*lst)

    
    def to_list(self) -> list:
        """
        Convert the Vector object to a list.

        This method returns a list containing the x, y, and z coordinates of the Vector object.

        Parameters:
        self (Vector): The Vector object to be converted.

        Returns:
        list: A list containing the x, y, and z coordinates of the Vector object.
        """
        return [self.x, self.y, self.z]

    
    def lerp(self, other: 'Vector', t: float) -> 'Vector':
        """
        Linearly interpolate between the current vector and another vector.

        This function calculates a new vector that lies on the line segment connecting the current vector and the other vector.
        The interpolation is determined by the parameter `t`, which ranges from 0 to 1.
        When `t` is 0, the function returns the current vector. When `t` is 1, the function returns the other vector.
        For any other value of `t`, the function returns a vector that lies on the line segment between the current vector and the other vector.

        Parameters:
        other (Vector): The other vector to interpolate with the current vector.
        t (float): A value between 0 and 1 that determines the interpolation point along the line segment connecting the current vector and the other vector.

        Returns:
        Vector: A new vector that is the result of linear interpolation between the current vector and the other vector.
        """
        return self * (1 - t) + other * t

    
    def rotate(self, angle: float, axis: 'Vector') -> 'Vector':
        """
        Rotate the current vector around a given axis by a specified angle.

        This function rotates the current vector by the given angle around the specified axis.
        The rotation is performed using the Rodrigues' rotation formula.

        Parameters:
        angle (float): The angle of rotation in degrees.
        axis (Vector): The axis around which to rotate the vector.

        Returns:
        Vector: A new vector that is the result of the rotation.
        """
        rad_angle = math.radians(angle)
        cos_theta = math.cos(rad_angle)
        sin_theta = math.sin(rad_angle)
        unit_axis = axis.normalize()

        return self.__mul__(cos_theta) + (unit_axis.cross_product(self)).__mul__(sin_theta) + unit_axis.__mul__((self.dot_product(unit_axis) * (1 - cos_theta)))

    def __add__(self, *other: 'Vector') -> 'Vector':
        """
        Add two or more Vector objects together.

        This method allows the addition of one or more Vector objects to the current Vector object.
        The addition is performed component-wise, i.e., the x-components, y-components, and z-components
        of the Vector objects are added together to form a new Vector object.

        Parameters:
        *other (Vector): One or more Vector objects to be added to the current Vector object.

        Returns:
        Vector: A new Vector object that is the result of adding the current Vector object and the other Vector objects.
        """
        x = self.x
        y = self.y
        z = self.z
        for vector in other:
            x += vector.x
            y += vector.y
            z += vector.z
        return Vector(x, y, z)

    
    def __sub__(self, *other: 'Vector') -> 'Vector':
        """
        Subtract one or more Vector objects from the current Vector object.

        This method allows the subtraction of one or more Vector objects from the current Vector object.
        The subtraction is performed component-wise, i.e., the x-components, y-components, and z-components
        of the Vector objects are subtracted together to form a new Vector object.

        Parameters:
        *other (Vector): One or more Vector objects to be subtracted from the current Vector object.

        Returns:
        Vector: A new Vector object that is the result of subtracting the current Vector object and the other Vector objects.
        """
        x = self.x
        y = self.y
        z = self.z
        for vector in other:
            x -= vector.x
            y -= vector.y
            z -= vector.z
        return Vector(x, y, z)

    
    def __mul__(self, scalar: float) -> 'Vector':
        """
        Multiply the current Vector object by a scalar.

        This method multiplies the x, y, and z components of the current Vector object by a scalar value.
        The multiplication is performed component-wise, resulting in a new Vector object.

        Parameters:
        scalar (float): The scalar to multiply the current Vector object by.

        Returns:
        Vector: A new Vector object that is the result of multiplying the current Vector object by the scalar.
        """

        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar: float) -> 'Vector':
        """
        Multiply the current Vector object by a scalar on the right side.

        This method allows a scalar to be multiplied to the current Vector object on the right side.
        The multiplication is performed component-wise, i.e., each component of the Vector object is
        multiplied by the scalar to form a new Vector object.

        Parameters:
        scalar (float): The scalar to multiply the current Vector object by.

        Returns:
        Vector: A new Vector object that is the result of multiplying the current Vector object by the scalar.
        """
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector':
        """
        Divide the current Vector object by a scalar.

        This method divides the x, y, and z components of the current Vector object by a scalar value.
        The division is performed component-wise, i.e., each component of the Vector object is divided
        by the scalar to form a new Vector object.

        Parameters:
        scalar (float): The scalar to divide the current Vector object by.

        Returns:
        Vector: A new Vector object that is the result of dividing the current Vector object by the scalar.
        """
        try:
            return self.__mul__(1 / scalar)
        except ZeroDivisionError:
            raise ZeroDivisionError("Cannot divide by zero")
    
    def __eq__(self, other: 'Vector') -> bool:
        """
        Check if the current Vector object is equal to another Vector object.

        This method compares the x, y, and z components of the current Vector object 
        with those of another Vector object to determine equality.

        Parameters:
        other (Vector): The other Vector object to compare with the current Vector object.

        Returns:
        bool: True if the current Vector object is equal to the other Vector object, False otherwise.
        """

        return self.x == other.x and self.y == other.y and self.z == other.z
    
    def __neg__(self) -> 'Vector':
        """
        Return a Vector object with the negation of the current Vector object's components.

        This method returns a new Vector object whose components are the negation of the current Vector object's components.

        Returns:
        Vector: A new Vector object with negated components.
        """
        return self.__mul__(-1)
    
    def __abs__(self) -> 'Vector':
        """
        Return a Vector object with absolute values of the current Vector object's components.

        This method returns a new Vector object whose components are the absolute values of the current Vector object's components.

        Returns:
        Vector: A new Vector object with absolute values of the components.
        """
        return Vector(abs(self.x), abs(self.y), abs(self.z))
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Vector object.
        
        Returns:
        str: A string representation of the Vector object in the format "Vector(x=..., y=..., z=...)"
        """
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"
    
    def __format__(self, format_spec: str) -> str:
        """
        Return a string representation of the Vector object with a specified number of decimal places.
        
        Parameters:
        format_spec (str): A string containing the desired number of decimal places, e.g., '.3f' for 3 decimal places.
        
        Returns:
        str: A string representation of the Vector object with the specified number of decimal places.
        
        return f"Vector({self.x:.{format_spec}f}, {self.y:.{format_spec}f}, {self.z:.{format_spec}f})"
        """
        return f"Vector(x={self.x:{format_spec}}, y={self.y:{format_spec}}, z={self.z:{format_spec}})"

    def __getitem__(self, index: int) -> float:
        """
        Get a specific component of the Vector object.

        This method allows the user to access a specific component (x, y, or z) of the Vector object
        by providing the index of the component.

        Parameters:
        index (int): The index of the component to be accessed. It should be 0 for x, 1 for y, and 2 for z.

        Returns:
        float: The value of the specified component.
        """
        return (self.x, self.y, self.z)[index]
        

    def __setitem__(self, index: int, value: float) -> None:
        """
        Set a new value for a specific component of the Vector object.

        This method allows the user to modify a specific component (x, y, or z) of the Vector object
        by providing the index of the component and the new value.

        Parameters:
        index (int): The index of the component to be modified. It should be 0 for x, 1 for y, and 2 for z.
        value (float): The new value to be assigned to the specified component.

        Returns:
        None
        """
        if index ==0:
            self.x = value
        elif index == 1:
            self.y = value
        elif index == 2:
            self.z = value
        else:
            raise IndexError("Index out of range")


    def __len__(self) -> int:
        """
        Get the number of components of the Vector object.

        This method returns the number of components in the Vector object, which is always 3.

        Returns:
        int: The number of components of the Vector object.
        """
        return 3
    
    def __bool__(self) -> bool:
        """
        Check if the Vector object is not zero.

        This method returns True if any of the x, y, or z components of the Vector object are not zero, and False otherwise.

        Returns:
        bool: True if the Vector object is not zero, False otherwise.
        """
        return self.x!= 0 or self.y!= 0 or self.z!= 0
    
    def __iter__(self):
        """
        Iterate over the components of the Vector object.

        This method allows the Vector object to be iterated over, yielding the x, y, and z components in that order.

        Returns:
        iterator: An iterator over the x, y, and z components of the Vector object.
        """
        return iter((self.x, self.y, self.z))

    def __lt__(self, other: 'Vector') -> bool:
        """
        Check if the current Vector object is less than another Vector object.

        This method compares the current Vector object with another Vector object
        and returns True if the current Vector object is less than the other Vector object,
        and False otherwise.

        Parameters:
        other (Vector): The other Vector object to compare with the current Vector object.

        Returns:
        bool: True if the current Vector object is less than the other Vector object, False otherwise.
        """
        return self.magnitude() < other.magnitude()

    
    def __str__(self) -> str:
        """
        Return a string representation of the Vector object in the format (x, y, z).

        Parameters:
        self (Vector): The Vector object for which the string representation is to be generated.

        Returns:
        str: A string representation of the Vector object in the format (x, y, z).
        """
        return f"({self.x}, {self.y}, {self.z})"

    
    __hash__ = None