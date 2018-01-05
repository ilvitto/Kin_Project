from array import array

import time
import scipy
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
from gap import gap

import sklearn.cluster as cluster
from sklearn.cluster import KMeans


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
            frame = Frame(face, faceHD, body, frame_i)
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

        needNewBlock = True
        blocks = 0

        for i, frame in enumerate(self._frames):
            # print i + 1, " of ", len(self._frames)
            if frame.isVeryGood(Descriptor.usedJoints):
                if(needNewBlock):
                    needNewBlock = False
                    blocks += 1
                currentFrames.append(frame)
                # temporaryDescriptor = self.processFrames(currentFrames)
                # classification = self.classify_people(allDescriptors, blocks)
                # classification = self.classify_people(descriptors + [temporaryDescriptor], blocks)
                updated = True
            elif not frame.isGood(Descriptor.usedJoints):
                if len(currentFrames) > 0 and updated:
                    descriptors.append(self.processFrames(currentFrames))
                    print "Classifying..."
                    classification = self.classify_people(descriptors, blocks)
                    print "...Classified"
                    updated = False
                # else:
                #     descriptors.append(CheckNFrames()._descriptorMedian)
                currentFrames = []
                needNewBlock = True

        # self.plot_feature(allDescriptors)

        # classify the number of people
        classification = self.classify_people(descriptors, blocks - 1)

        print classification.cluster_centers_
        print classification.labels_

        # compute centroids of clusters
        people = classification.cluster_centers_

        # self.plot_feature(descriptors)
        self.save_people(filename="learned_people.txt", people=people)

    def processFrames(self, frames):
        return CheckNFrames(frames)._descriptorMedian

    def classify_people(self, descriptors, clustersNumber):
        people = []
        X = np.stack(descriptors[i].getFeatures() for i in range(len(descriptors)))

        print "Finding best K..."
        gaps, s_k, K = gap.gap_statistic(X, refs=None, B=10, K=range(1, clustersNumber + 1), N_init=10)
        bestKValue = gap.find_optimal_k(gaps, s_k, K)
        print "Optimal K -> ", bestKValue, " of ", clustersNumber

        classification = KMeans(n_clusters=bestKValue, random_state=0).fit(X)

        # TODO: SUPERVISED SAVED RESULTS
        found = False
        threshold = 15  # DA SISTEMARE
        best = 1
        print classification.labels_
        # plt.plot(range(1, len(errors)+1), np.array(errorsPerCent))
        # plt.show()
        return classification

    # TODO: intracluster distance for all
    def intraClusterDistance(self, classification, X):
        error = 0
        max_error = 0
        for k in range(len(classification.cluster_centers_)):
            for i, x in enumerate(X):
                if (classification.labels_[i] == k):
                    error = np.amax([scipy.spatial.distance.euclidean(x, classification.cluster_centers_[k]), error])
            max_error = np.amax([error, max_error])

        return max_error

    def plot_feature(self, blocks):
        for block in blocks:
            d1 = []
            d2 = []
            for descriptor in block:
                d1.append(descriptor._chestColor[0])
                d2.append(descriptor._chestColor[1])
            plt.plot(d1, d2, 'o')
        plt.show()
        # plt.plot(range(0, len(descriptors)), np.array(d1))

    def save_people(self, filename, people):
        output = file(filename, "w")
        for person in people:
            output.write(person)
            output.write("\n")
        output.close()
