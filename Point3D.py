from math import sqrt
from math import pow as power


class Point3D:
    def __init__(self, x, y, z):
        self._x = x
        self._y = y
        self._z = z

    def getX(self):
        return self._x

    def getY(self):
        return self._y

    def getZ(self):
        return self._z

    def distance(self, other):
        return sqrt(power(self.getX() - other.getX(), 2) + power(self.getY() - other.getY(), 2) + power(
            self.getZ() - other.getZ(), 2))

