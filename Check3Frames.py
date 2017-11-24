import Descriptor

class Check3Frames:

    def __init__(self, frames):
        self._descriptor = self.getDescriptor(frames)

    def getDescriptor(self, frames):
        descriptors = []
        for frame in frames:
            descriptors.append(Descriptor.Descriptor(frame))
        return self.average(descriptors)

    def average(self, descriptors):
        return