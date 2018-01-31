import numpy as np
import os
import shutil
import sys

import cv2
import pylab
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

dataset_folder = "./dataset"
savedFrames_folder = './savedFrames'
GAP = "gap"
THRESH = "thresh"

OUTPUT_FILE = "learned_people.txt"
NUM_CLUSTERS = "num_clusters.txt"


class Kin:
    def __init__(self):
        self._infoVideo = None
        self._frames = None
        self._filename = None
        self._allDescriptors = None

    def run(self, filename=None, colors=False, method=GAP, printDetails=True):
        classification = None
        oldClusters = 0

        # check choherence between stored images and dataset file
        self.checkCoherence()

        # load dataset
        if (os.path.isfile(OUTPUT_FILE) and os.path.isfile(NUM_CLUSTERS)):

            self._allDescriptors, oldClusters = self.load_people(OUTPUT_FILE, NUM_CLUSTERS)
            print "Dataset: ", len(self._allDescriptors)
            print "Number of people: ", oldClusters

        # self.load_all_datasets()
        self._filename = filename
        if filename is not None:
            self.load(dataset_folder + "/" + filename + "/body_and_face.mat")
            # get descriptors
            descriptors = self.getDescriptors()

            print "Processing..."
            all_classifications = []
            for descriptor in descriptors:

                #Save relevant frame for each median descriptor
                self.saveRelevantFrame(descriptor)

                #First case for incompatible size
                if self._allDescriptors is None:
                    self._allDescriptors = [descriptor.getFeatures()]
                else:
                    if colors:
                        self._allDescriptors = np.concatenate(
                            (self._allDescriptors, [descriptor.getFeatures() + descriptor.getColorFeature()]))
                    else:
                        self._allDescriptors = np.concatenate((self._allDescriptors, [descriptor.getFeatures()]))

                if len(self._allDescriptors) == 1:
                    classification = KMeans(n_clusters=1, random_state=0).fit(self._allDescriptors).labels_
                elif len(self._allDescriptors) < 3:
                    #STESSO VIDEO
                    best_match = 0
                    for i, oldDesc in enumerate(self._allDescriptors[:-1]):
                        if(descriptor.featuresDistance(descriptor.getFeatures(), oldDesc) <
                               descriptor.featuresDistance(descriptor.getFeatures(), self._allDescriptors[best_match])):
                            best_match = i
                    if descriptor.isNearToEachFeatures(self._allDescriptors[best_match]):
                        classification = np.zeros(len(self._allDescriptors))
                        classification[-1] = 1
                        classification[best_match] = 1

                    # if(descriptors.index(descriptor) == 1):
                    #     print 'Using color too...'
                    #     if descriptor.isNearToEachFeatures(self._allDescriptors[0], descriptors[0].getColorFeature()):
                    #         classification = KMeans(n_clusters=1, random_state=0).fit(self._allDescriptors).labels_
                    #     else:
                    #         classification = KMeans(n_clusters=2, random_state=0).fit(self._allDescriptors).labels_
                    # else:
                    #     if descriptor.isNearToEachFeatures(self._allDescriptors[0]):
                    #         classification = KMeans(n_clusters=1, random_state=0).fit(self._allDescriptors).labels_
                    #     else:
                    #         classification = KMeans(n_clusters=2, random_state=0).fit(self._allDescriptors).labels_
                else:
                    classification = self.classify(self._allDescriptors, oldClusters, colors=colors, method=method)

                all_classifications.append(classification)
                oldClusters = len(set(classification))




            if len(descriptors) > 0:
                classification = self.classify(self._allDescriptors, oldClusters,  colors=colors, method=method, printDetails=printDetails)

            # SHOW VIDEO OR SHOW IMAGES
            if self._allDescriptors is not None:
                if self.askVideo():
                    self.showVideo(self._allDescriptors, descriptors, all_classifications)
                else:
                    self.showImagesResults(self._allDescriptors, descriptors, all_classifications)

            if len(descriptors) > 0:
                if self.ask_supervised():
                    X = np.stack(self._allDescriptors[i] for i in range(len(self._allDescriptors)))
                    self.save_people(filename=OUTPUT_FILE, people=X)
                    self.save_people(filename=NUM_CLUSTERS, people=np.array([len(set(classification))]))
                else:
                    self.removeLastNImages(len(descriptors))



            return self._allDescriptors, classification
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

    def getDescriptors(self):
        print "Number of frame: " + str(len(self._frames))

        descriptors = []

        currentFrames = []
        updated = False

        needNewBlock = True
        blocks = 0

        for i, frame in enumerate(self._frames):
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

    def ask_supervised(self):
        sys.stdout.write("Do you want to save data classification? (y/n)")
        s = raw_input().lower()
        if s == "y" or s == "Yes":
            return True
        return False

    def askVideo(self):
        sys.stdout.write("Do you want to show the video?")
        s = raw_input().lower()
        if s == "y" or s == "Yes":
            return True
        return False

    def classify(self, descriptors, oldClusters, colors=False, method=GAP, printDetails=True):
        if (method == GAP):
            return self.classify_people_with_gap(descriptors, oldClusters, colors, printDetails)
        else:
            return self.classify_people_with_threshold(descriptors, colors, printDetails)

    def classify_people_with_threshold(self, descriptors, colors=False, printDetails=True):
        if printDetails: print "Using Elbow method with threshold..."
        X = np.stack(descriptors[i] for i in range(len(descriptors)))
        errors = []
        errorsPerCent = []
        classifications = []
        for i in range(len(descriptors)):
            classification = KMeans(n_clusters=i + 1, random_state=0).fit(X).labels_
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
        if printDetails: print "Labels -> ", classifications[best - 1]
        # plt.plot(range(1, len(errors)+1), np.array(errorsPerCent))
        # plt.show()
        return classifications[best - 1]

    def classify_people_with_gap(self, descriptors, oldClusters, colors=False, printDetails=True):
        if printDetails: print "Using GAP method..."
        X = np.stack(descriptors[i] for i in range(len(descriptors)))
        if printDetails: print "Finding best K..."
        ks = []
        for i in range(5):
            gaps, s_k, K = gap.gap_statistic(X, refs=None, B=10, K=range(np.maximum(1,oldClusters-2), np.minimum(len(descriptors + 1), oldClusters + 3)), N_init=10)
            bestKValue = gap.find_optimal_k(gaps, s_k, K)
            ks.append(bestKValue)
        (values, counts) = np.unique(ks, return_counts=True)
        ind = np.argmax(counts)
        bestKValue = values[ind]
        if printDetails: print "Optimal K -> ", bestKValue, " of ", len(descriptors), " chosen between ", ks
        classification = KMeans(n_clusters=bestKValue, random_state=0).fit(X).labels_
        if printDetails: print "Labels -> ", classification
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

    def nearestPointToCentroid(self, classification, targetLabel, X):
        if targetLabel in classification.labels_:  # esiste almeno un altro elemento assegnato
            centroid_of_cluster = classification.cluster_centers_[targetLabel]
            top = None
            minError = scipy.spatial.distance.euclidean(X[0], centroid_of_cluster)
            for i, x in enumerate(X[:-1]):
                if (classification.labels_[i] == targetLabel):
                    error = scipy.spatial.distance.euclidean(x, centroid_of_cluster)
                    if error <= minError:
                        minError = error
                        top = i
            return top
        return None

    #Restituisce il punto piu vicino a X[point_index] tra i punti dello stesso cluster
    def nearestPointSameCluster(self, classification, X, point_index):
        top = None
        minError = None
        for i, x in enumerate(X[:-1]):
            if (classification[i] == classification[point_index]):
                error = scipy.spatial.distance.euclidean(x, X[point_index])
                if minError is None or error <= minError:
                    minError = error
                    top = i
        return top

    # Restituisce il punto piu vicino a X[point_index] tra i punti dello stesso cluster
    def nearestPointDifferentCluster(self, classification, X, point_index):
        top = None
        minError = None
        for i, x in enumerate(X[:-1]):
            if (classification[i] != classification[point_index]):
                error = scipy.spatial.distance.euclidean(x, X[point_index])
                if minError is None or error <= minError:
                    minError = error
                    top = i
        return top

    #Restituisce il punto piu vicino
    def nearestPoint(self, X, point_index):
        top = None
        minError = scipy.spatial.distance.euclidean(X[0], X[point_index])
        for i, x in enumerate(X[:-1]):
            error = scipy.spatial.distance.euclidean(x, X[point_index])
            if error <= minError:
                minError = error
                top = i
        return top

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

    def load_people(self, filename, filename2):
        db = np.loadtxt(filename)
        oldClusters = int(np.loadtxt(filename2))
        return db, oldClusters

    def checkCoherence(self):
        if (not os.path.isfile(OUTPUT_FILE)):
            if(len(os.listdir(savedFrames_folder)) > 0):
                self.emptyDatabase()
            return
        num_dataset = len(np.loadtxt(OUTPUT_FILE))
        num_images = len(os.listdir(savedFrames_folder))
        if num_dataset != num_images:
            print "Found corrupted database!"
            print "Dataset: ", num_dataset
            print "Images: ", num_images
            if num_images > num_dataset:
                sys.stdout.write("Do you want I try to repair it? (y/n)")
                s = raw_input().lower()
                if s == "y" or s == "Yes":
                    self.removeLastNImages(num_images - num_dataset)
                    return
            sys.stdout.write("Do you want to delete all data? (y/n)")
            s = raw_input().lower()
            if s == "y" or s == "Yes":
                self.emptyDatabase()
            else:
                sys.exit()
        return

    def removeLastNImages(self, n):
        names = [name for name in os.listdir(savedFrames_folder) if os.path.isfile(os.path.join(savedFrames_folder, name))]
        names.sort()
        names = names[-n:]
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

    def showDoubleImage(self, img1, img2, title1='Image 1', title2='Image 2'):
        f = pylab.figure()
        f.set_size_inches(10,6)
        arr = np.asarray(img1)
        f.add_subplot(1, 2, 1)
        pylab.axis("off")
        pylab.imshow(arr)
        pylab.title(title1)
        arr = np.asarray(img2)
        f.add_subplot(1, 2, 2)
        pylab.axis("off")
        pylab.imshow(arr)
        pylab.title(title2)
        pylab.show()

    def showImage(self, img, title='Figure'):
        arr = np.asarray(img)
        pylab.axis("off")
        pylab.imshow(arr)
        pylab.title(title)
        pylab.show()

    def emptyDatabase(self):
        if os.path.isfile(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        if os.path.isfile(NUM_CLUSTERS):
            os.remove(NUM_CLUSTERS)
        for name in os.listdir(savedFrames_folder):
            os.remove(savedFrames_folder + '/' + name)

    def showImagesResults(self, allFeaturesDescriptors, newDescriptors, classifications):

        #PRINT DOUBLE IMAGES
        descriptor_number = 0
        for classification in classifications:
            targetLabel = classification[-1]

            X = allFeaturesDescriptors[:len(allFeaturesDescriptors)-len(newDescriptors) + 1 + descriptor_number]
            oldImageNumber = None
            unique, counts = np.unique(classification, return_counts=True)
            #Founded a corrispondence
            if dict(zip(unique, counts))[targetLabel] > 1:
                oldImageNumber = self.nearestPointSameCluster(classification, X, len(X)-1)

            newImage = cv2.imread(dataset_folder + "/" + self._filename + "/rgbReg_frames/" + newDescriptors[0].realImageName(newDescriptors[descriptor_number]._frame._frame_number) + ".jpg")
            if oldImageNumber is not None:
                oldImage = cv2.imread(savedFrames_folder + "/" + newDescriptors[0].realImageName(oldImageNumber + 1) + ".jpg")
                self.showDoubleImage(oldImage, newImage, 'Recognition from database', 'New frame')
            else:
                self.showImage(newImage, 'New person classified')
            descriptor_number += 1

    def showVideo(self, allFeaturesDescriptors, newDescriptors, classifications):
        frame_number = 0
        recognized = False
        descriptor_number = 0
        cap = cv2.VideoCapture(dataset_folder + '/' + self._filename + '/rgbReg_video.mj2')
        while cap.isOpened() and frame_number < len(self._frames) and descriptor_number < len(classifications):
            ret, frame = cap.read()

            X = allFeaturesDescriptors[:len(allFeaturesDescriptors)-len(newDescriptors) + 1 + descriptor_number]
            if frame_number >= newDescriptors[descriptor_number]._start_frame and \
                            frame_number <= newDescriptors[descriptor_number]._end_frame:
                recognized = True
                if self._frames[frame_number]._face is not None:
                    cv2.rectangle(frame, (self._frames[frame_number]._face._boundingBox._lu.asArray()),
                              (self._frames[frame_number]._face._boundingBox._rb.asArray()), (0, 255, 0), 3)
                unique, counts = np.unique(classifications[descriptor_number], return_counts=True)
                #Founded a match
                if dict(zip(unique, counts))[classifications[descriptor_number][-1]] > 1:
                    #i_best = self.nearestPointToCentroid(classifications[descriptor_number], classifications[descriptor_number].labels_[-1], X)
                    i_best = self.nearestPointSameCluster(classifications[descriptor_number],X,len(X)-1)
                    #TODO: compute accuracy method
                    nearest_diff = self.nearestPointDifferentCluster(classifications[descriptor_number],X,len(X)-1)
                    if i_best is not None and nearest_diff is not None:
                        accuracy = round((1 - (newDescriptors[0].featuresDistance(X[i_best], X[-1]) / (2 * newDescriptors[0].featuresDistance(X[nearest_diff], X[-1])) ))*100,2)
                    else:
                        accuracy = '!'
                    #accuracy = 100-abs((newDescriptors[0].featuresDistance(X[i_best], X[-1])-Descriptor.euclideanThreshold)*100)

                    cv2.putText(frame, "Recognized "+str(accuracy)+'%', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    image = cv2.imread(savedFrames_folder + "/" + newDescriptors[0].realImageName(i_best+1) + ".jpg")
                    cv2.imshow('Person recognized', image)

                #New person identified
                else:
                    # cv2.putText(frame, "New person", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    image = cv2.imread(dataset_folder + "/new-person.jpg")
                    cv2.imshow('Person recognized', image)
            else:
                #First exit of the subject
                if recognized:
                    recognized = False
                    descriptor_number += 1
                image = cv2.imread(dataset_folder + "/unknown.jpg")
                cv2.imshow('Person recognized', image)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1
        cap.release()
        cv2.destroyAllWindows()