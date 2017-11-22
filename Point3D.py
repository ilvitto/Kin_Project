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