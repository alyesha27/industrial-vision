
import time
import cv2


class VideoSource:
    def __init__(self, source=0, reconnect_sec=2):
        self.source = source
        self.reconnect_sec = reconnect_sec
        self.cap = None

    def open(self):
        # Закрываем старое подключение, если оно было
        if self.cap is not None:
            self.cap.release()

        # Открываем камеру или видеофайл
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть источник видео: {self.source}"
            )

    def read(self):
        # Если камера не открыта, пробуем подключиться
        if self.cap is None or not self.cap.isOpened():
            self.open()

        # Получаем кадр
        success, frame = self.cap.read()

        if success and frame is not None:
            return frame

        # Если кадр получить не удалось, пробуем переподключиться
        print("Не удалось получить кадр. Повторное подключение...")
        time.sleep(self.reconnect_sec)
        self.open()

        success, frame = self.cap.read()

        if not success or frame is None:
            raise RuntimeError(
                "Видеопоток недоступен после повторного подключения"
            )

        return frame

    def close(self):
        # Освобождаем камеру
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FPSMeter:
    def __init__(self, window=30):
        self.times = []
        self.window = window

    def tick(self):
        # Запоминаем время получения кадра
        now = time.perf_counter()
        self.times.append(now)

        # Храним только последние измерения
        if len(self.times) > self.window:
            self.times.pop(0)

        if len(self.times) < 2:
            return 0.0

        elapsed = self.times[-1] - self.times[0]

        if elapsed <= 0:
            return 0.0

        return (len(self.times) - 1) / elapsed
