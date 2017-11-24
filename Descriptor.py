from matplotlib import pyplot as plt

class Descriptor:

    def __init__(self, frame):
        if(frame._body is not None):
            self._height = None
            self._shoulderDistance = self.shoulderDistance(frame._body._points)
            self._shoulderDistance2 = self.shoulderDistance2(frame._body._points)
            self._leftArm = None
            self._rightArm = None
            self._leftLeg = None
            self._rightLeg = None
        else:
            self._height = 0
            self._shoulderDistance = 0
            self._shoulderDistance2 = 0
            self._leftArm = 0
            self._rightArm = 0
            self._leftLeg = 0
            self._rightLeg = 0

        #estrai descrittori dal frame

    def shoulderDistance(self, points):
        return points[4].distance(points[20]) + points[20].distance(points[8])

    def shoulderDistance2(self, points):
        return points[4].distance(points[8])