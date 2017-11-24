class Descriptor:

    def __init__(self, frame):
        distances = []
        if(frame._body is not None):
            self._height = None
            self._shoulderDistance = self.shoulderDistance(frame._body._points)
            print self._shoulderDistance
            self._leftArm = None
            self._rightArm = None
            self._leftLeg = None
            self._rightLeg = None
        else:
            print "???"

        #estrai descrittori dal frame

    def shoulderDistance(self, points):
        return points[4].distance(points[20]) + points[20].distance(points[8]);