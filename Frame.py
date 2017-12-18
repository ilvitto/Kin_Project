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
