class Rect2D:
    def __init__(self, leftUp, rightBottom):
        self._lu = leftUp
        self._rb = rightBottom

    def getLeftUp(self):
        return self._lu

    def getRightBottom(self):
        return self._rb
