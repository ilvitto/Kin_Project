import Point3D
import numpy as np
import matplotlib.pyplot as plt
from Joint import Joint

class Body:
    def __init__(self, frame):
        self.assignJoints(frame)
        self._leftHandState = self.getLeftHandState(frame)
        self._rightHandState = self.getRightHandState(frame)

    def assignJoints(self, frame):
        self._joints = []
        positions = self.getPosition(frame)
        orientations = self.getOrientation(frame)
        trackingStates = self.getTrackingState(frame)

        for i in range(positions.item().shape[1]):
            point = Point3D.Point3D(positions.item()[0][i],positions.item()[1][i], positions.item()[2][i])
            orientation = Joint(0,0,0).getRotation3D(orientations.item()[0][i],orientations.item()[1][i],orientations.item()[2][i],orientations.item()[3][i])
            #Point3D.Point3D(positions.item()[0][i],positions.item()[1][i], positions.item()[2][i])
            trackingState = trackingStates.item()[i]
            self._joints.append(Joint(point,orientation,trackingState))

    def getJoint(self, j):
        return self._joints[j]

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