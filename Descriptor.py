from matplotlib import pyplot as plt
from Joint import Joint

class Descriptor:

    def __init__(self, frame):
        if(frame._body is not None):
            self._frame = frame
            self._joints = frame._body._joints
            self._shoulderDistance = self.getShoulderDistance()
            self._shoulderDistance2 = self.getDirectShoulderDistance()
            self._leftArmLong = self.getLeftArmLong()
            self._rightArmLong = self.getRightArmLong()
            self._leftLegLong = self.getLeftLegLong()
            self._rightLegLong = self.getRightLegLong()
            self._height = self.getHeight()
        else:
            self._frame = None
            self._joints = None
            self._shoulderDistance = None
            self._shoulderDistance2 = None
            self._leftArmLong = None
            self._rightArmLong = None
            self._leftLegLong = None
            self._rightLegLong = None
            self._height = None

    def jointsDistance(self, j1, j2):
        if self._joints is not None and self._joints[j1].isTracked() and self._joints[j2].isTracked():
            return self._joints[j1].distance(self._joints[j2])
        return None

    def getShoulderDistance(self):
        d1 = self.jointsDistance(Joint.ShoulderLeft, Joint.SpineShoulder)
        d2 = self.jointsDistance(Joint.ShoulderRight, Joint.SpineShoulder)
        return d1 + d2 if (d1 is not None and d2 is not None) else None

    def getDirectShoulderDistance(self):
        d1 = self.jointsDistance(Joint.ShoulderLeft, Joint.ShoulderRight)
        return d1 if (d1 is not None) else None

    def getLeftArmLong(self):
        d1 = self.jointsDistance(Joint.ShoulderLeft, Joint.ElbowLeft)
        d2 = self.jointsDistance(Joint.ElbowLeft, Joint.WristLeft)
        return d1 + d2 if (d1 is not None and d2 is not None) else None

    def getRightArmLong(self):
        d1 = self.jointsDistance(Joint.ShoulderRight, Joint.ElbowRight)
        d2 = self.jointsDistance(Joint.ElbowRight, Joint.WristRight)
        return d1 + d2 if (d1 is not None and d2 is not None) else None

    #12->13->14
    def getLeftLegLong(self):
        d1 = self.jointsDistance(Joint.HipLeft, Joint.KneeLeft)
        d2 = self.jointsDistance(Joint.KneeLeft, Joint.AnkleLeft)
        return d1 + d2 if (d1 is not None and d2 is not None) else None

    # 16->17->18
    def getRightLegLong(self):
        d1 = self.jointsDistance(Joint.HipRight, Joint.KneeRight)
        d2 = self.jointsDistance(Joint.KneeRight, Joint.AnkleRight)
        return d1 + d2 if (d1 is not None and d2 is not None) else None

    # 0->1->20
    def getChestLong(self):
        d1 = self.jointsDistance(Joint.SpineBase, Joint.SpineMid)
        d2 = self.jointsDistance(Joint.SpineMid, Joint.SpineShoulder)
        return d1 + d2 if (d1 is not None and d2 is not None) else None

    # 20->2->3
    def getHeadLong(self):
        d1 = self.jointsDistance(Joint.SpineShoulder, Joint.Neck)
        d2 = self.jointsDistance(Joint.Neck, Joint.Head)
        return d1 + d2 if (d1 is not None and d2 is not None) else None


    def getHeight(self):
        head = self.getHeadLong()
        chest = self.getChestLong()
        leftLeg = self.getLeftLegLong()
        rightLeg = self.getRightLegLong()
        return head + chest + (leftLeg + rightLeg) / 2 if (head is not None and chest is not None and rightLeg is not None and leftLeg is not None) else None