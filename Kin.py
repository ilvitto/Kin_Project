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
        for frame in self._frames:
            if frame.isVeryGood(Descriptor.usedJoints):
                currentFrames.append(frame)
                temporaryDescriptor = self.processFrames(currentFrames)
                classification = self.classify_people(descriptors + [temporaryDescriptor])
                updated = True
            elif not frame.isGood(Descriptor.usedJoints):
                if len(currentFrames) > 0 and updated:
                    descriptors.append(self.processFrames(currentFrames))
                    updated = False
                # else:
                #     descriptors.append(CheckNFrames()._descriptorMedian)
                currentFrames = []

        # classify the number of people
        classification = self.classify_people(descriptors)

        print classification.cluster_centers_
        print classification.labels_

        # compute centroids of clusters
        people = classification.cluster_centers_

        # self.plot_feature(descriptors)
        self.save_people(filename="learned_people.txt", people=people)

    def processFrames(self, frames):
        return CheckNFrames(frames)._descriptorMedian

    def classify_people(self, descriptors):
        people = []
        X = np.stack(descriptors[i].getFeatures() for i in range(len(descriptors)))

        gaps, s_k, K = gap.gap_statistic(X, refs=None, B=10, K=range(1, len(descriptors) + 1), N_init=10)
        bestKValue = gap.find_optimal_k(gaps, s_k, K)
        print "Optimal K -> ", bestKValue

        np.set_printoptions(precision=3)
        bounding_box = np.stack(self.bounding_box(X)).transpose()

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

    def bounding_box(self, X):
        return X.min(0), X.max(0)

    # def gap_statistic(X):
    #     (xmin, xmax), (ymin, ymax) = bounding_box(X)
    #     # Dispersion for real distribution
    #     ks = range(1, 10)
    #     Wks = zeros(len(ks))
    #     Wkbs = zeros(len(ks))
    #     sk = zeros(len(ks))
    #     for indk, k in enumerate(ks):
    #         mu, clusters = find_centers(X, k)
    #         Wks[indk] = np.log(Wk(mu, clusters))
    #         # Create B reference datasets
    #         B = 10
    #         BWkbs = zeros(B)
    #         for i in range(B):
    #             Xb = []
    #             for n in range(len(X)):
    #                 Xb.append([random.uniform(xmin, xmax),
    #                            random.uniform(ymin, ymax)])
    #             Xb = np.array(Xb)
    #             mu, clusters = find_centers(Xb, k)
    #             BWkbs[i] = np.log(Wk(mu, clusters))
    #         Wkbs[indk] = sum(BWkbs) / B
    #         sk[indk] = np.sqrt(sum((BWkbs - Wkbs[indk]) ** 2) / B)
    #     sk = sk * np.sqrt(1 + 1 / B)
    #     return (ks, Wks, Wkbs, sk)


    def plot_feature(self, descriptors):
        d1 = []
        for descriptor in descriptors:
            if descriptor._shoulderDistance is not None:
                d1.append(descriptor._shoulderDistance)
            else:
                d1.append(0)

        plt.plot(range(0, len(descriptors)), np.array(d1))
        plt.show()

    def save_people(self, filename, people):
        output = file(filename, "w")
        for person in people:
            output.write(person)
            output.write("\n")
        output.close()
