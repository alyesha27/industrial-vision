from collections import deque


class TemporalConfirm:
    """Подтверждение по правилу N из M: событие истинно,
    если детекция была минимум в N из последних M кадров."""

    def __init__(self, n=3, m=5):
        self.n = n
        self.window = deque(maxlen=m)

    def update(self, detected):
        self.window.append(bool(detected))
        return len(self.window) == self.window.maxlen and sum(self.window) >= self.n