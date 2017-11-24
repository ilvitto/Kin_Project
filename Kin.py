from array import array

import scipy.io as sio
import numpy as np
import InfoVideo
import Frame
import Face
import FaceHD
import Body

class Kin:
    def __init__(self):
        self._infoVideo = None
        self._frames = None

    def load(self, filename):
        self._processMatlabData(sio.loadmat(filename, squeeze_me=True))

    def _processMatlabData(self, data):
        self._infoVideo = InfoVideo.InfoVideo(data['info_video'])
        self._frames = []
        for frame_i in range(self._infoVideo._frames):
            face = None
            faceHD = None
            body = None
            if data['face_gallery'][frame_i].size == 1:
                face = Face.Face(data['face_gallery'][frame_i])
            if data['HD_face_gallery'][frame_i].size == 1:
                faceHD = FaceHD.FaceHD(data['HD_face_gallery'][frame_i])
            if data['body_gallery'][frame_i].size == 1:
                body = Body.Body(data['body_gallery'][frame_i])
            frame = Frame.Frame(face, faceHD, body)
            self._frames.append(frame)
