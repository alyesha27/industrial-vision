
class ROI:
    def __init__(self, x1, y1, x2, y2):
        # Координаты прямоугольной зоны
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def contains(self, x, y):
        # Проверяем, находится ли точка внутри зоны
        return (
            self.x1 <= x <= self.x2
            and self.y1 <= y <= self.y2
        )
