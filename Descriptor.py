from matplotlib import pyplot as plt

class Descriptor:
    # JointType_SpineBase = 1;
    # JointType_SpineMid = 2;
    # JointType_Neck = 3;
    # JointType_Head = 4;
    # JointType_ShoulderLeft = 5;
    # JointType_ElbowLeft = 6;
    # JointType_WristLeft = 7;
    # JointType_HandLeft = 8;
    # JointType_ShoulderRight = 9;
    # JointType_ElbowRight = 10;
    # JointType_WristRight = 11;
    # JointType_HandRight = 12;
    # JointType_HipLeft = 13;
    # JointType_KneeLeft = 14;
    # JointType_AnkleLeft = 15;
    # JointType_FootLeft = 16;
    # JointType_HipRight = 17;
    # JointType_KneeRight = 18;
    # JointType_AnkleRight = 19;
    # JointType_FootRight = 20;
    # JointType_SpineShoulder = 21;
    # JointType_HandTipLeft = 22;
    # JointType_ThumbLeft = 23;
    # JointType_HandTipRight = 24;
    # JointType_ThumbRight = 25;
    # JointType_Count = 25;

    def __init__(self, frame):
        if(frame._body is not None):
            self._frame = frame
            self._joints = frame._body._points
            self._shoulderDistance = self.getShoulderDistance(frame)
            self._shoulderDistance2 = self.getShoulderDistance2(frame)
            self._leftArmLong = self.getLeftArmLong(frame)
            self._rightArmLong = self.getRightArmLong(frame)
            self._leftLegLong = self.getLeftLegLong(frame)
            self._rightLegLong = self.getRightLegLong(frame)
            self._height = self.getHeight(frame)
        else:
            self._frame = None
            self._joints = None
            self._shoulderDistance = 0
            self._shoulderDistance2 = 0
            self._leftArmLong = 0
            self._rightArmLong = 0
            self._leftLegLong = 0
            self._rightLegLong = 0
            self._height = 0

    def getShoulderDistance(self, frame):
        return frame._body._points[4].distance(frame._body._points[20]) + frame._body._points[20].distance(frame._body._points[8])

    def getShoulderDistance2(self, frame):
        return frame._body._points[4].distance(frame._body._points[8])

    # 4->5->6
    def getLeftArmLong(self, frame):
        if self.isTrackedPoint(frame, 4) and self.isTrackedPoint(frame, 5) and self.isTrackedPoint(frame, 6):
            return self.distance(frame._body._points[4],frame._body._points[5])+self.distance(frame._body._points[5],frame._body._points[6])
        return 0

    #8->9->10
    def getRightArmLong(self, frame):
        if self.isTrackedPoint(frame, 8) and self.isTrackedPoint(frame, 9) and self.isTrackedPoint(frame, 10):
            return self.distance(frame._body._points[8], frame._body._points[9]) + self.distance(frame._body._points[9], frame._body._points[10])
        return 0

    #12->13->14
    def getLeftLegLong(self, frame):
        if self.isTrackedPoint(frame, 12) and self.isTrackedPoint(frame, 13) and self.isTrackedPoint(frame, 14):
            return self.distance(frame._body._points[12], frame._body._points[13]) + self.distance(frame._body._points[13], frame._body._points[14])
        return 0

    # 16->17->18
    def getRightLegLong(self, frame):
        if self.isTrackedPoint(frame, 16) and self.isTrackedPoint(frame, 17) and self.isTrackedPoint(frame, 18):
            return self.distance(frame._body._points[16], frame._body._points[17]) + self.distance(frame._body._points[17], frame._body._points[18])
        return 0

    # 0->1->20
    def getChestLong(self, frame):
        if self.isTrackedPoint(frame, 0) and self.isTrackedPoint(frame, 1) and self.isTrackedPoint(frame, 20):
            self.distance(frame._body._points[0], frame._body._points[1]) + self.distance(frame._body._points[1], frame._body._points[20])
        return 0

    # 20->2->3
    def getHeadLong(self, frame):
        if self.isTrackedPoint(frame, 20) and self.isTrackedPoint(frame, 2) and self.isTrackedPoint(frame, 3):
            self.distance(frame._body._points[20], frame._body._points[2]) + self.distance(frame._body._points[2], frame._body._points[3])
        return 0


    def getHeight(self, frame):
        if self.getRightLegLong(frame) > 0 and self.getRightArmLong(frame) > 0 and self.getChestLong(frame) > 0 and self.getHeadLong(frame) > 0:
            return self.getRightLegLong(frame) + self.getRightArmLong(frame) + self.getChestLong(frame) + self.getHeadLong(frame)
        return 0

    # TrackingState: NotTracked = 0, Inferred = 1, or Tracked = 2
    def isTrackedPoint(self, frame, i):
        return frame._body._points[i]._trackingState != 0

    def shoulderDistance(self, frame):
        return frame._body._points[4].distance(frame._body._points[20]) + frame._body._points[20].distance(frame._body._points[8])

    def distance(self, p1, p2):
        return p1.distance(p2)