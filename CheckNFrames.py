from Descriptor import Descriptor
from Frame import Frame
import numpy as np

class CheckNFrames:

    def __init__(self, frames = [], filename = None):
        self._frames = frames
        self._filename = filename
        # self._descriptorAvg = self.getDescriptorAvg(frames)
        self._descriptorMedian = self.getDescriptorMedian(frames, filename)
        self._descriptorMedian._frame = self.nearestFrameToMedianDescriptor()


    # def getDescriptorAvg(self, frames):
    #     descriptors = []
    #     for frame in frames:
    #         descriptors.append(Descriptor(frame))
    #     return self.average(descriptors)

    def getDescriptorMedian(self, frames, filename):
        descriptors = []
        for frame in frames:
            descriptors.append(Descriptor(frame, filename))
        return self.median(descriptors)
        #return Descriptor(frames[-1])
    # def average(self, descriptors):
    #     shoulderDistances = []
    #     shoulderDirectDistances = []
    #     leftArmLongs = []
    #     rightArmLongs = []
    #     leftLegLongs = []
    #     rightLegLongs = []
    #     heights = []
    #     descriptor = Descriptor(self._frames[0])
    #     for descr in descriptors:
    #         shoulderDistances.append(descr._shoulderDistance) if descr._shoulderDistance is not None else None
    #         shoulderDirectDistances.append(descr._shoulderDirectDistance) if descr._shoulderDirectDistance is not None else None
    #         leftArmLongs.append(descr._leftArmLong) if descr._leftArmLong is not None else None
    #         rightArmLongs.append(descr._rightArmLong) if descr._rightArmLong is not None else None
    #         leftLegLongs.append(descr._leftLegLong) if descr._leftLegLong is not None else None
    #         rightLegLongs.append(descr._rightLegLong) if descr._rightLegLong is not None else None
    #         heights.append(descr._height) if descr._height is not None else None
    #     descriptor._shoulderDistance = np.mean(shoulderDistances) if len(shoulderDistances) > 0 else None
    #     descriptor._leftArmLong = np.mean(leftArmLongs) if len(leftArmLongs) > 0 else None
    #     descriptor._rightArmLong = np.mean(rightArmLongs) if len(rightArmLongs) > 0 else None
    #     descriptor._leftLegLong = np.mean(leftLegLongs) if len(leftLegLongs) > 0 else None
    #     descriptor._rightLegLong = np.mean(rightLegLongs) if len(rightLegLongs) > 0 else None
    #     descriptor._height = np.mean(heights) if len(heights) > 0 else None
    #     return descriptor

    def median(self, descriptors):
        shoulderDistances = []
        shoulderDirectDistances = []
        leftArmLongs = []
        rightArmLongs = []
        leftLegLongs = []
        rightLegLongs = []
        heights = []
        clLefts = []
        clRights = []
        chestColor = []
        descriptor = Descriptor()
        for descr in descriptors:
            shoulderDistances.append(descr._shoulderDistance) if descr._shoulderDistance is not None else None
            shoulderDirectDistances.append(
                descr._shoulderDirectDistance) if descr._shoulderDirectDistance is not None else None
            leftArmLongs.append(descr._leftArmLong) if descr._leftArmLong is not None else None
            rightArmLongs.append(descr._rightArmLong) if descr._rightArmLong is not None else None
            leftLegLongs.append(descr._leftLegLong) if descr._leftLegLong is not None else None
            rightLegLongs.append(descr._rightLegLong) if descr._rightLegLong is not None else None
            heights.append(descr._height) if descr._height is not None else None
            clLefts.append(descr._clavicleLeft) if descr._clavicleLeft is not None else None
            clRights.append(descr._clavicleRight) if descr._clavicleRight is not None else None
            chestColor.append(descr._chestColor) if descr._chestColor is not None else None
        descriptor._filename = self._filename
        descriptor._shoulderDistance = np.median(shoulderDistances) if len(shoulderDistances) > 0 else None
        descriptor._leftArmLong = np.median(leftArmLongs) if len(leftArmLongs) > 0 else None
        descriptor._rightArmLong = np.median(rightArmLongs) if len(rightArmLongs) > 0 else None
        descriptor._leftLegLong = np.median(leftLegLongs) if len(leftLegLongs) > 0 else None
        descriptor._rightLegLong = np.median(rightLegLongs) if len(rightLegLongs) > 0 else None
        descriptor._height = np.median(heights) if len(heights) > 0 else None
        descriptor._clavicleLeft = np.median(clLefts) if len(clLefts) > 0 else None
        descriptor._clavicleRight = np.median(clRights) if len(clRights) > 0 else None
        descriptor._chestColor = np.mean(chestColor, axis=(0)) if len(chestColor) > 0 else None
        return descriptor

    def nearestFrameToMedianDescriptor(self):
        minDistance = self._descriptorMedian.descriptorDistance(Descriptor(self._frames[0], self._filename))
        bestFrame = self._frames[0]
        for frame in self._frames:
            distance = self._descriptorMedian.descriptorDistance(Descriptor(frame, self._filename))
            if distance < minDistance:
                minDistance = distance
                bestFrame = frame
        return bestFrame