import numpy as np
import os
import sys

import cv2
import scipy
import scipy.io as sio
from gap import gap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from Body import Body
from CheckNFrames import CheckNFrames
from Descriptor import Descriptor
from Face import Face
from FaceHD import FaceHD
from Frame import Frame
from InfoVideo import InfoVideo
import pylab
import matplotlib.cm as cm
import shutil

dataset_folder = "./dataset"
savedFrames_folder = './savedFrames'
GAP = "gap"
THRESH = "thresh"

OUTPUT_FILE = "learned_people.txt"


class Kin:
    def __init__(self):
        self._infoVideo = None
        self._frames = None
        self._filename = None

    def run(self, filename=None, colors=False, method=GAP):
        # load dataset
        if (os.path.isfile(OUTPUT_FILE)):
            allDescriptors = self.load_people(OUTPUT_FILE)
        else:
            allDescriptors = []
        # self.load_all_datasets()
        self._filename = filename
        if filename is not None:
            self.load(dataset_folder + "/" + filename + "/body_and_face.mat")
            # get descriptors
            descriptors = self.getDescriptors()

            print "Processing..."
            for descriptor in descriptors:
                #Save relevant frame for each median descriptor
                self.saveRelevantFrame(descriptor)

                #First case for incompatible size
                if allDescriptors == []:
                    allDescriptors = [descriptor.getFeatures()]
                else:
                    if colors:
                        allDescriptors = np.concatenate(
                            (allDescriptors, [descriptor.getFeatures() + descriptor.getColorFeature()]))
                    else:
                        allDescriptors = np.concatenate((allDescriptors, [descriptor.getFeatures()]))

                classification = self.classify(allDescriptors, colors=colors, method=method)
                print classification.labels_

                # TODO: Choose the best image from saved Frames of index-cluster
                # find nearest point to the center
                oldImageNumber = None
                targetLabel = classification.labels_[-1]
                X = allDescriptors[:-1]
                if targetLabel in classification.labels_[:-1]:#esiste almeno un altro elemento assegnato
                    centroid_of_cluster = classification.cluster_centers_[targetLabel]
                    top = 0
                    minError = scipy.spatial.distance.euclidean(X[top], centroid_of_cluster)
                    for i, x in enumerate(X):
                        if (classification.labels_[i] == targetLabel):
                            error = scipy.spatial.distance.euclidean(x, centroid_of_cluster)
                            if error < minError:
                                minError = error
                                top = i
                    oldImageNumber = top

                #OLD METHOD
                #targetLabel = classification.labels_[-1]
                #oldImageNumber = None
                # for index, label in enumerate(classification.labels_[:-1]):
                #
                #     if label == targetLabel:
                #         oldImageNumber = index

                newImage = cv2.imread(dataset_folder + "/" + filename + "/rgbReg_frames/" + descriptor.realImageName(
                                    descriptor._frame._frame_number) + ".jpg")
                if oldImageNumber is not None:
                    oldImage = cv2.imread("./savedFrames/" + descriptor.realImageName(oldImageNumber + 1) + ".jpg")
                    self.showDoubleImage(oldImage, newImage)
                else:
                    self.showImage(newImage)

            classification = self.classify(allDescriptors, colors=colors, method=method)

            if self.ask_supervised():
                X = np.stack(allDescriptors[i] for i in range(len(allDescriptors)))
                self.save_people(filename=OUTPUT_FILE, people=X)
            else:
                self.removeLastNImages(len(descriptors))

            return allDescriptors, classification
        return [], None

    def load_all_datasets(self):
        for filename in os.listdir(dataset_folder):
            self.load(dataset_folder + '/' + filename + "/body_and_face.mat")

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

    # TODO    if len(descriptors) == 1 -> median descriptor
    # TODO    if len(descriptors) == 2 -> classify with thresholds per features/limit error/range values
    # TODO    if len(descriptors) > 3 -> KMeans
    def getDescriptors(self):
        print "Number of frame: " + str(len(self._frames))

        descriptors = []

        currentFrames = []
        updated = False

        needNewBlock = True
        blocks = 0

        for i, frame in enumerate(self._frames):
            # print i + 1, " of ", len(self._frames)
            if frame.isVeryGood(Descriptor.usedJoints):
                if (needNewBlock):
                    needNewBlock = False
                    blocks += 1
                currentFrames.append(frame)
                updated = True
            elif not frame.isGood(Descriptor.usedJoints):
                if len(currentFrames) > 0 and updated:
                    descriptors.append(self.processFrames(currentFrames))
                    updated = False
                currentFrames = []
                needNewBlock = True

        if descriptors == []:
            print "No descriptors founded in the video #" + self._filename
        else:
            print "Found " + str(blocks) + " descriptors"

        return descriptors

    # TODO: Check color invariant
    def checkIfColorIsRelevant(self, classification, classification_with_color):
        # check if there are equal centroids excluding colors (last 3 features)

        pass

    def processFrames(self, frames):
        return CheckNFrames(frames, self._filename)._descriptorMedian

    # TODO: SUPERVISED SAVED RESULTS
    def ask_supervised(self):
        sys.stdout.write("Do you want to save data classification? (y/n)")
        s = raw_input().lower()
        if s == "y" or s == "Yes":
            return True
        return False

    def classify(self, descriptors, colors=False, method=GAP):
        if (method == GAP):
            return self.classify_people_with_gap(descriptors, colors)
        else:
            return self.classify_people_with_threshold(descriptors, colors)

    def classify_people_with_threshold(self, descriptors, printDetails=True):
        if printDetails: print "Using Elbow method with threshold..."
        X = np.stack(descriptors[i] for i in range(len(descriptors)))
        errors = []
        errorsPerCent = []
        classifications = []
        for i in range(len(descriptors)):
            classification = KMeans(n_clusters=i + 1, random_state=0).fit(X)
            error = self.intraClusterDistanceCentroids(classification, X)
            classifications.append(classification)
            errors.append(error)

        for error in errors:
            errorsPerCent.append((max(errors) - error) / (max(errors) - min(errors)) * 100)

        found = False
        threshold = 15  # TODO: Choose a better threshold
        best = 1
        for i in range(1, len(errorsPerCent)):
            if errorsPerCent[i] - errorsPerCent[i - 1] > threshold:
                found = True
            if found and errorsPerCent[i] - errorsPerCent[i - 1] < threshold:
                best = i
                break
        if printDetails: print "Optimal K -> ", best
        if printDetails: print "Labels -> ", classifications[best - 1].labels_
        # plt.plot(range(1, len(errors)+1), np.array(errorsPerCent))
        # plt.show()
        return classifications[best - 1]

    def classify_people_with_gap(self, descriptors, printDetails=True):
        if printDetails: print "Using GAP method..."
        X = np.stack(descriptors[i] for i in range(len(descriptors)))
        if printDetails: print "Finding best K..."
        gaps, s_k, K = gap.gap_statistic(X, refs=None, B=10, K=range(1, len(descriptors) + 1), N_init=10)
        bestKValue = gap.find_optimal_k(gaps, s_k, K)
        if printDetails: print "Optimal K -> ", bestKValue, " of ", len(descriptors)
        classification = KMeans(n_clusters=bestKValue, random_state=0).fit(X)
        if printDetails: print "Labels -> ", classification.labels_
        return classification

    def intraClusterDistanceCentroids(self, classification, X):
        error = 0
        max_error = 0
        for k in range(len(classification.cluster_centers_)):
            for i, x in enumerate(X):
                if (classification.labels_[i] == k):
                    error = np.amax([scipy.spatial.distance.euclidean(x, classification.cluster_centers_[k]), error])
            max_error = np.amax([error, max_error])
        return max_error

    def intraClusterDistanceAll(self, classification, X):
        error = 0
        max_error = 0
        for k in range(len(classification.cluster_centers_)):
            for j, y in enumerate(X):
                for i, x in enumerate(X):
                    if (classification.labels_[i] == k and classification.labels_[j] == k):
                        error = np.amax([scipy.spatial.distance.euclidean(x, y), error])
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

    def save_people(self, filename, people):
        np.savetxt(filename, people)

    def load_people(self, filename):
        return np.loadtxt(filename)

    def removeLastNImages(self, n):
        names = [name for name in os.listdir(savedFrames_folder) if os.path.isfile(os.path.join(savedFrames_folder, name))][-n:]
        for name in names:
            os.remove(savedFrames_folder + '/' + name)

    def saveRelevantFrame(self, descriptor):
        rgbReg_frames = dataset_folder + '/' + descriptor._filename + '/rgbReg_frames'
        if not os.path.exists(savedFrames_folder):
            os.makedirs(savedFrames_folder)
        counting = len([name for name in os.listdir(savedFrames_folder) if os.path.isfile(os.path.join(savedFrames_folder, name))])
        str_frame_number = descriptor.realImageName(descriptor._frame._frame_number)
        new_str_frame_number = descriptor.realImageName(counting + 1)
        source = rgbReg_frames + '/' + str_frame_number + '.jpg'
        destination = savedFrames_folder + '/' + new_str_frame_number + '.jpg'
        shutil.copyfile(source, destination)

    def showDoubleImage(self, img1, img2):
        f = pylab.figure()
        arr = np.asarray(img1)
        f.add_subplot(1, 2, 1)
        pylab.imshow(arr, cmap=cm.Greys_r)
        pylab.title('From database')
        arr = np.asarray(img2)
        f.add_subplot(1, 2, 2)
        pylab.imshow(arr, cmap=cm.Greys_r)
        pylab.title('New image')
        pylab.show()

    def showImage(self, img):
        arr = np.asarray(img)
        pylab.imshow(arr, cmap=cm.Greys_r)
        pylab.title('New person classified')
        pylab.show()