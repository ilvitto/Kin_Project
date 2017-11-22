from FaceProperties import FaceProperties
from Point2D import Point2D
from Point3D import Point3D
from Rect2D import Rect2D
from Rotation3D import Rotation3D


class Face:
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
        points = []
        for joint in data['FacePoints'].item():
            points.append(Point2D(joint[0], joint[1]))
        return points

    def _loadPoints3D(self, data):
        points = []
        for joint in data['FacePoints3D'].item():
            points.append(Point3D(joint[0], joint[1], joint[2]))
        return points

    def _loadRotation(self, data):
        Rotation3D(data["FaceRotation"].item()[0], data["FaceRotation"].item()[1], data["FaceRotation"].item()[2])

    def _loadProperties(self, data):
        properties = data['FaceProperties'].item()
        return FaceProperties(properties[0], properties[1], properties[2], properties[3], properties[4], properties[5],
                              properties[6], properties[7])
