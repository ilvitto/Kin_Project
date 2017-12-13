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
        for i in range(blockSize - 1, len(self._frames)):
            frames = []
            for j in range(i - blockSize + 1, i+1):
                frames.append(self._frames[j])
            descriptorAvg = CheckNFrames(frames)._descriptorAvg
            descriptorMedian = CheckNFrames(frames)._descriptorMedian
            descriptorsAvg.append(descriptorAvg)
            descriptorsMedian.append(descriptorMedian)
            print "Overlap #"+str(i)+": "
            print "Average: ",
            descriptorAvg.showDescriptor()
            print "Median: ",
            descriptorMedian.showDescriptor()
            print "orientation:",
            if Descriptor(self._frames[i])._joints is not None:
                print Descriptor(self._frames[i])._joints[Joint.ShoulderLeft].getRotation3D(),
                print Descriptor(self._frames[i])._joints[Joint.ShoulderRight].getRotation3D(),

        distances1 = []
        distances2 = []
        distances3 = []
        print "Starting plotting: "
        print len(descriptorsAvg),len(descriptorsMedian), len(self._frames)

        for i in range(len(self._frames) - blockSize + 1):
            distance = descriptorsAvg[i]._shoulderDistance
            distance2 = descriptorsMedian[i]._shoulderDistance
            distance3 = Descriptor(self._frames[i])._shoulderDistance

            distances1.append(distance if distance is not None else 0)
            distances2.append(distance2 if distance2 is not None else 0)
            distances3.append(distance3 if distance3 is not None else 0)
            # distances2.append(Descriptor(self._frames[i]).getChestLong())
            # distances4.append(Descriptor(self._frames[i]).getRightLegLong())

        # plt.plot(np.array(distances1))
        plt.plot(range(0,len(self._frames) - blockSize + 1),sig.medfilt(np.array(distances1), 9),range(0,len(self._frames) - blockSize + 1),sig.medfilt(np.array(distances2), 9),range(0,len(self._frames) - blockSize + 1),sig.medfilt(np.array(distances3), 9))

        # plt.plot(sig.medfilt(np.array(distances2), 9))
        plt.show()

        return descriptorsAvg, descriptorMedian
