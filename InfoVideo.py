class InfoVideo:
    def __init__(self, data):
        self._duration = data['Duration']
        self._frames = data['Frames']
        self._fps = data['fps']