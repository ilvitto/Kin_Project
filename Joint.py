class Joint:

    def __init__(self, position, orientation, trackingState):
        self._position = position
        self._orientation = orientation
        self._trackingState = trackingState

    def isTracked(self):
        return self._trackingState > 0

    def distance(self, joint):
        return self._position.distance(joint._position)