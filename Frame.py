from Joint import Joint
import numpy as np

class Frame:
    def __init__(self, face=None, faceHD=None, body=None):
        self._face = face
        self._faceHD = faceHD
        self._body = body

    def getBody(self):
        return self._body

    def isGood(self, usedJoints):
        if self._body is None:
            return False
        for j in usedJoints:
            if not self._body.getJoint(j).isTracked():
                return False
        return True

    def isVeryGood(self, usedJoints):
        shoulderLeftThreshold = 20
        shoulderRightThreshold = 20
        if not self.isGood(usedJoints):
            return False

        if np.abs(self._body.getJoint(Joint.ShoulderLeft)._orientation[1]) > shoulderLeftThreshold or \
                        np.abs(self._body.getJoint(Joint.ShoulderRight)._orientation[1]) > shoulderRightThreshold:
            return False
        return True
