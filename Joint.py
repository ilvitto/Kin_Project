import numpy as np

class Joint:
    SpineBase = 0
    SpineMid = 1
    Neck = 2
    Head = 3
    ShoulderLeft = 4
    ElbowLeft = 5
    WristLeft = 6
    HandLeft = 7
    ShoulderRight = 8
    ElbowRight = 9
    WristRight = 10
    HandRight = 11
    HipLeft = 12
    KneeLeft = 13
    AnkleLeft = 14
    FootLeft = 15
    HipRight = 16
    KneeRight = 17
    AnkleRight = 18
    FootRight = 19
    SpineShoulder = 20
    HandTipLeft = 21
    ThumbLeft = 22
    HandTipRight = 23
    ThumbRight = 24

    def __init__(self, position, orientation, trackingState):
        self._position = position
        self._orientation = orientation
        self._trackingState = trackingState

    def isTracked(self):
        return self._trackingState > 0

    def distance(self, joint):
        return self._position.distance(joint._position)

    def getRotation3D(self, q1=None,q2=None,q3=None,q4=None):
        print self._orientation
        if(q1 is None and q2 is None and q3 is None and q4 is None):

            q1 = self._orientation[0]
            q2 = self._orientation[1]
            q3 = self._orientation[2]
            q4 = self._orientation[3]
        q = np.array((q1,q2,q3,q4))
        yaw = np.arctan2(2 * (q[0] * q[3] - q[1] * q[2]),
                         1 - 2 * (q[2] ** 2 + q[3] ** 2))
        pitch = np.arcsin(2 * (q[0] * q[2] + q[3] * q[1]))
        roll = np.arctan2(2 * (q[0] * q[1] - q[2] * q[3]),
                          1 - 2 * (q[1] ** 2 + q[2] ** 2))
        return np.array([yaw, pitch, roll])