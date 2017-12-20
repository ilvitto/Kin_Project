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
        for frame in self._frames:
            if frame.isVeryGood(Descriptor.usedJoints):
                currentFrames.append(frame)
                descriptors.append(self.processFrames(currentFrames))
            else:
                descriptors.append(CheckNFrames()._descriptorMedian)
                if not frame.isGood(Descriptor.usedJoints):
                    print "> " + str(len(currentFrames)) + " <"
                    currentFrames = []

        d1 = []
        for descriptor in descriptors:
            if(descriptor._leftArmLong is not None):
                d1.append(descriptor._leftArmLong)
            else:
                d1.append(0)

        plt.plot(range(0, len(descriptors)), np.array(d1))
        plt.show()



    def processFrames(self, frames):
        return CheckNFrames(frames)._descriptorMedian

