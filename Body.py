import Point3D
import numpy as np
import matplotlib.pyplot as plt

class Body:
    def __init__(self, frame):
        self.getPoints(frame)
        self._leftHandState = self.getLeftHandState(frame)
        self._rightHandState = self.getRightHandState(frame)

    def getPoints(self, frame):
        self._points = []
        positions = self.getPosition(frame)
        orientations = self.getOrientation(frame)
        trackingStates = self.getTrackingState(frame)

        for i in range(positions.item().shape[1]):
            point = Point3D.Point3D(positions.item()[0][i],positions.item()[1][i], positions.item()[2][i])
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
