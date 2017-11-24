import numpy as np
from Rect2D import Rect2D
from Point2D import Point2D

class FaceHD:
    def __init__(self, faceData):
        self._boundingBox = self._loadBoundingBox(faceData)
        self._rotation = self._loadRotation(faceData)
        self._headPivot = self.loadHeadPivot(faceData)
        self._animationUnits = self.loadAnimationUnits(faceData)
        self._shapeUnits = self.loadShapeUnits(faceData)
        self._faceModel = self.loadFaceModel(faceData)

    def _loadBoundingBox(self, data):
        data = data['FaceBox']
        return Rect2D(Point2D(data.item()[0], data.item()[1]), Point2D(data.item()[2], data.item()[3]))

    def _loadRotation(self, data):
        return data['FaceRotation']

    def loadHeadPivot(self, data):
        return data['HeadPivot']

    def loadAnimationUnits(self, data):
        return data['AnimationUnits']

    def loadShapeUnits(self, data):
        return data['ShapeUnits']

    def loadFaceModel(self, data):
        return data['FaceModel']