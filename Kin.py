import scipy.io as sio
import InfoVideo

class Kin:
    def __init__(self):
        self._data = None

    def load(self, filename):
        self._data = self._processMatlabData(sio.loadmat(filename, squeeze_me=True))

    def _processMatlabData(self, data):
        infoVideo = data['info_video']
        return InfoVideo.InfoVideo(infoVideo)

    def getInfoVideo(self):
        return self._data