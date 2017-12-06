from Descriptor import Descriptor
import numpy as np


class CheckNFrames:

    def __init__(self, frames):
        self._frames = frames
        self._descriptor = self.getDescriptor(frames)

    def getDescriptor(self, frames):
        descriptors = []
        for frame in frames:
            descriptors.append(Descriptor(frame))
        return self.average(descriptors)

    def average(self, descriptors):
        shoulderDistances = []
        shoulderDirectDistances = []
        leftArmLongs = []
        rightArmLongs = []
        leftLegLongs = []
        rightLegLongs = []
        heights = []
        descriptor = Descriptor(self._frames[0])
        for descr in descriptors:
            shoulderDistances.append(descr._shoulderDistance) if descr._shoulderDistance is not None else None
            shoulderDirectDistances.append(descr._shoulderDirectDistance) if descr._shoulderDirectDistance is not None else None
            leftArmLongs.append(descr._leftArmLong) if descr._leftArmLong is not None else None
            rightArmLongs.append(descr._rightArmLong) if descr._rightArmLong is not None else None
            leftLegLongs.append(descr._leftLegLong) if descr._leftLegLong is not None else None
            rightLegLongs.append(descr._rightLegLong) if descr._rightLegLong is not None else None
            heights.append(descr._height) if descr._height is not None else None
        descriptor._shoulderDistance = np.mean(shoulderDistances) if len(shoulderDistances) > 0 else None
        descriptor._leftArmLong = np.mean(leftArmLongs) if len(leftArmLongs) > 0 else None
        descriptor._rightArmLong = np.mean(rightArmLongs) if len(rightArmLongs) > 0 else None
        descriptor._leftLegLong = np.mean(leftLegLongs) if len(leftLegLongs) > 0 else None
        descriptor._rightLegLong = np.mean(rightLegLongs) if len(rightLegLongs) > 0 else None
        descriptor._height = np.mean(heights) if len(heights) > 0 else None
        return descriptor