class Rotation3D:
    def __init__(self, pitch, yaw, roll):
        self._pitch = pitch
        self._yaw = yaw
        self._roll = roll

    def getPitch(self):
        return self._pitch

    def getYaw(self):
        return self._yaw

    def getRoll(self):
        return self._roll