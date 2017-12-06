from math import sqrt
from math import pow as power

class Point2D:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def getX(self):
        return self._x

    def getY(self):
        return self._y

    def distance(self, other):
        return sqrt(power(self.getX() - other.getX(), 2) + power(self.getY() - other.getY(), 2))

    def __str__(self):
        return "(" + str(self._x) + "," + str(self._y) + ")"