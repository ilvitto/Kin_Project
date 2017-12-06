import Point3D
import numpy as np
import matplotlib.pyplot as plt
from Joint import Joint

class Body:
    def __init__(self, frame):
        self.getJoints(frame)
        self._leftHandState = self.getLeftHandState(frame)
        self._rightHandState = self.getRightHandState(frame)

    def getJoints(self, frame):
        self._joints = []
        positions = self.getPosition(frame)
        orientations = self.getOrientation(frame)
        trackingStates = self.getTrackingState(frame)

        for i in range(positions.item().shape[1]):
            point = Point3D.Point3D(positions.item()[0][i],positions.item()[1][i], positions.item()[2][i])
            orientation = self.fromQuaternion(orientations.item()[0][i],orientations.item()[1][i],orientations.item()[2][i],orientations.item()[3][i])
            if(i== Joint.Head):
                print orientation
            #Point3D.Point3D(positions.item()[0][i],positions.item()[1][i], positions.item()[2][i])
            trackingState = trackingStates.item()[i]
            self._joints.append(Joint(point,orientation,trackingState))


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

    def fromQuaternion(self, q1,q2,q3,q4):
        q = np.array((q1,q2,q3,q4))
        yaw = np.arctan2(2 * (q[0] * q[3] - q[1] * q[2]),
                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        pitch = np.arcsin(2 * (q[0] * q[2] + q[3] * q[1]))
        roll = np.arctan2(2 * (q[0] * q[1] - q[2] * q[3]),
                          1 - 2 * (q[1] ** 2 + q[2] ** 2))
        return np.array([yaw, pitch, roll])