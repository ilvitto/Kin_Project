import Point3D
import numpy as np

class Body:
    def __init__(self, frame):
        self._points = self.getPoints(frame)
        self._leftHandState = self.getLeftHandState(frame)
        self._rightHandState = self.getRightHandState(frame)

    def getPoints(self, frame):
        self._points = []
        positions = self.getPosition(frame)
        orientations = self.getOrientation(frame)
        trackingStates = self.getTrackingState(frame)
        for i in range(positions.size):
            point = Point3D.Point3D(positions.item()[i],orientations.item()[i], trackingStates.item()[i])
            self._points.append(point)

    def getPosition(self, frame):
        return frame['Position']

    def getOrientation(self, frame):
        return frame['Orientation']

    def getTrackingState(self, frame):
        return frame['TrackingState']

    def getLeftHandState(self, frame):
        return frame['LeftHandState']

    def getRightHandState(self, frame):
        return frame['RightHandState']
