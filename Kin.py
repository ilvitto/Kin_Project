from array import array

import scipy.io as sio
import numpy as np
import InfoVideo
import Frame
import Face
import FaceHD
import Body
from Check3Frames import Check3Frames
from Descriptor import Descriptor
from matplotlib import pyplot as plt

class Kin:
    def __init__(self):
        self._infoVideo = None
        self._frames = None

    def load(self, filename):
        self._processMatlabData(sio.loadmat(filename, squeeze_me=True))

    def _processMatlabData(self, data):
        self._infoVideo = InfoVideo.InfoVideo(data['info_video'])
        self._frames = []
        for frame_i in range(self._infoVideo._frames):
            face = None
            faceHD = None
            body = None
            if data['face_gallery'][frame_i].size == 1:
                face = Face.Face(data['face_gallery'][frame_i])
            if data['HD_face_gallery'][frame_i].size == 1:
                faceHD = FaceHD.FaceHD(data['HD_face_gallery'][frame_i])
            if data['body_gallery'][frame_i].size == 1:
                body = Body.Body(data['body_gallery'][frame_i])
            frame = Frame.Frame(face, faceHD, body)
            self._frames.append(frame)

    def getDescriptors(self):
        descriptors = []
        blockSize = 3
        for i in range(blockSize - 1, len(self._frames)):
            frames = []
            for j in range(i - blockSize + 1, i):
                frames.append(self._frames[j])
            descriptors.append(Check3Frames(frames)._descriptor)

        distances1 = []
        distances2 = []
        for i in range(len(self._frames)):
            distances1.append(Descriptor(self._frames[i])._shoulderDistance)
            distances2.append(Descriptor(self._frames[i])._shoulderDistance2)
        plt.plot(distances1)
        plt.plot(distances2)
        plt.show()

        return descriptors
