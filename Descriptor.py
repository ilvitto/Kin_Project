from matplotlib import pyplot as plt
from Joint import Joint
import cv2
import scipy
import numpy as np

class Descriptor:
    usedJoints = [Joint.ShoulderLeft, Joint.SpineShoulder, Joint.SpineMid, Joint.SpineBase, Joint.ShoulderRight,
                  Joint.ElbowLeft, Joint.ElbowRight,
                  Joint.WristLeft, Joint.WristRight, Joint.Head, Joint.Neck, Joint.HipLeft, Joint.HipRight,
                  Joint.KneeLeft, Joint.KneeRight, Joint.AnkleLeft, Joint.AnkleRight, Joint.Neck]

    def __init__(self, frame=None, filename=None):
        if (frame is not None and frame._body is not None):
            self._frame = frame
            self._joints = frame._body._joints
            self._filename = filename
            self._shoulderDistance = self.getShoulderDistance()
            self._shoulderDirectDistance = self.getDirectShoulderDistance()
            self._leftArmLong = self.getLeftArmLong()
            self._rightArmLong = self.getRightArmLong()
            self._leftLegLong = self.getLeftLegLong()
            self._rightLegLong = self.getRightLegLong()
            self._height = self.getHeight()
            self._clavicleLeft = self.getClavicleLeft()
            self._clavicleRight = self.getClavicleRight()

            self._chestColor = self.getChestColor()
        else:
            self._frame = None
            self._joints = None
            self._filename = None
            self._shoulderDistance = None
            self._shoulderDirectDistance = None
            self._leftArmLong = None
            self._rightArmLong = None
            self._leftLegLong = None
            self._rightLegLong = None
            self._height = None
            self._clavicleLeft = None
            self._clavicleRight = None

            self._chestColor = None

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

    # 12->13->14
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

    def getClavicleLeft(self):
        return self.jointsDistance(Joint.ShoulderLeft, Joint.Neck)

    def getClavicleRight(self):
        return self.jointsDistance(Joint.ShoulderRight, Joint.Neck)

    def getHeight(self):
        head = self.getHeadLong()
        chest = self.getChestLong()
        leftLeg = self.getLeftLegLong()
        rightLeg = self.getRightLegLong()
        return head + chest + (leftLeg + rightLeg) / 2 if (
            head is not None and chest is not None and rightLeg is not None and leftLeg is not None) else None

    # TODO: Remove dependece with dataset
    def getChestColor(self):
        if self._joints is not None and self._joints[Joint.SpineMid].isTracked():
            str_frame_number = self.realImageName(self._frame._frame_number)

            rgbImage = cv2.imread('./dataset/' + self._filename + '/rgbReg_frames/' + str_frame_number + '.jpg')
            indexImage = cv2.imread('./dataset/' + self._filename + '/bodyIndex_frames/' + str_frame_number + ".png")
            indexImage = cv2.cvtColor(indexImage, cv2.COLOR_BGR2GRAY)
            rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)

            ret, mask = cv2.threshold(indexImage, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            img2_fg = cv2.bitwise_and(rgbImage, rgbImage, mask=mask_inv)

            (r, g, b, _) = cv2.mean(img2_fg, mask=mask_inv)

        return np.array([r, g, b]) / 255

    def realImageName(self, frame_number):
        if frame_number < 10:
            return "000" + str(frame_number)
        elif frame_number < 100:
            return "00" + str(frame_number)
        elif frame_number < 1000:
            return "0" + str(frame_number)
        else:
            return str(frame_number)

    def showDescriptor(self):
        if self.isEmpty():
            print " Empty Descriptor"
        else:
            print ""
            print "     Shoulder distance: ",
            if self._shoulderDistance is not None:
                print self._shoulderDistance
            print "     Shoulder direct distance: ",
            if self._shoulderDirectDistance is not None:
                print self._shoulderDirectDistance
            print "     Left arm long: ",
            if self._leftArmLong is not None:
                print self._leftArmLong
            print "     Right arm long: ",
            if self._rightArmLong is not None:
                print self._rightArmLong
            print "     Left leg long: ",
            if self._leftLegLong is not None:
                print self._leftLegLong
            print "     Right leg long: ",
            if self._rightLegLong is not None:
                print self._rightLegLong
            print "     Height: ",
            if self._height is not None:
                print self._height

    def isEmpty(self):
        return True if self._shoulderDistance is None and self._shoulderDirectDistance is None and self._leftArmLong is None \
                       and self._rightArmLong is None and self._leftArmLong is None and self._rightArmLong is None and self._height is None else False

    # _shoulderDistance, _shoulderDirectDistance, _leftArmLong, _rightArmLong, _leftLegLong, _rightLegLong, _height, _clavicleLeft, _clavicleRight
    def getFeatures(self):
        return [self._shoulderDistance, self._leftArmLong, self._rightArmLong \
            , self._leftLegLong, self._rightLegLong, self._height, self._clavicleLeft, self._clavicleRight]

    def getColorFeature(self):
        return [self._chestColor[0], self._chestColor[1], self._chestColor[2]]

    def descriptorDistance(self, descriptor):
        return scipy.spatial.distance.euclidean(self.getFeatures(), descriptor.getFeatures())
