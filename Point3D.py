from math import sqrt
from math import pow as power


class Point3D:
    def __init__(self, position, orientation, trackingState):
        self._position = position
        self._orientation = orientation
        self._trackingState = trackingState

    def getX(self):
        return self._position

    def getY(self):
        return self._orientation

    def getZ(self):
        return self._trackingState

    def distance(self, other):
        return sqrt(power(self.getX() - other.getX(), 2) + power(self.getY() - other.getY(), 2) + power(
            self.getZ() - other.getZ(), 2))
