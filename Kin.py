from array import array

import scipy.io as sio
import scipy.signal as sig
import numpy as np
from InfoVideo import InfoVideo
from Frame import Frame
from Face import Face
from FaceHD import FaceHD
from Body import Body
from Joint import Joint
from CheckNFrames import CheckNFrames
from Descriptor import Descriptor
from matplotlib import pyplot as plt
import math


class Kin:
    def __init__(self):
        self._infoVideo = None
        self._frames = None

    def load(self, filename):
        self._processMatlabData(sio.loadmat(filename, squeeze_me=True))

    def _processMatlabData(self, data):
        self._infoVideo = InfoVideo(data['info_video'])
        self._frames = []
        for frame_i in range(self._infoVideo._frames):
            face = None
            faceHD = None
            body = None
            if data['face_gallery'][frame_i].size == 1:
                face = Face(data['face_gallery'][frame_i])
            if data['HD_face_gallery'][frame_i].size == 1:
                faceHD = FaceHD(data['HD_face_gallery'][frame_i])
            if data['body_gallery'][frame_i].size == 1:
                body = Body(data['body_gallery'][frame_i])
            frame = Frame(face, faceHD, body)
            self._frames.append(frame)

    def getDescriptors(self):
        descriptorsAvg = []
        descriptorsMedian = []
        blockSize = 5
        print "Number of frame: " + str(len(self._frames))
        print "Block size: " + str(blockSize) + "\n"

        descriptors = []
        currentFrames = []
        updated = False
        for frame in self._frames:
            if frame.isVeryGood(Descriptor.usedJoints):
                currentFrames.append(frame)
                updated = True
            elif not frame.isGood(Descriptor.usedJoints):
                if len(currentFrames) > 0 and updated:
                    descriptors.append(self.processFrames(currentFrames))
                    updated = False
                else:
                    descriptors.append(CheckNFrames()._descriptorMedian)
                currentFrames = []

        d1 = []
        d2 = []
        for descriptor in descriptors:
            if descriptor._shoulderDistance is not None and descriptor._leftLegLong is not None and descriptor._rightLegLong is not None:
                d1.append(descriptor._shoulderDistance / (np.mean([descriptor._leftLegLong, descriptor._rightLegLong])))
            else:
                d1.append(0)

            if (descriptor._shoulderDistance is not None):
                d2.append(
                    descriptor._shoulderDistance)
            else:
                d2.append(0)

        plt.plot(range(0, len(descriptors)), np.array(d1))
        plt.plot(range(0, len(descriptors)), np.array(d2))
        plt.show()

    def processFrames(self, frames):
        return CheckNFrames(frames)._descriptorMedian
