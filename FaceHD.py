import numpy as np
from Rect2D import Rect2D
from Point2D import Point2D

class FaceHD:
    def __init__(self, faceData):
        self._boundingBox = self._loadBoundingBox(faceData)
        self._points = self._loadPoints(faceData)
        self._points3D = self._loadPoints3D(faceData)
        self._rotation = self._loadRotation(faceData)
        self._properties = self._loadProperties(faceData)

    def _loadBoundingBox(self, data):
        data = data['FaceBox']
        return Rect2D(Point2D(data.item()[0], data.item()[1]), Point2D(data.item()[2], data.item()[3]))

    def _loadPoints(self, data):
        return data['FacePoints']

    def _loadPoints3D(self, data):
        return data['FacePoints3D']

    def _loadRotation(self, data):
        return data['FaceRotation']

    def _loadProperties(self, data):
        return data['FaceProperties']