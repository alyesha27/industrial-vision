
import time
import psutil


class Metrics:
    def __init__(self):
        self.process = psutil.Process()
        self.process.cpu_percent()

    def start_timer(self):
        return time.perf_counter()

    def get_inference_time(self, start):
        return (time.perf_counter() - start) * 1000

    def get_cpu(self):
        # cpu_percent() считает по ядрам, поэтому нормируем на их число
        return self.process.cpu_percent() / psutil.cpu_count()

    def get_ram(self):
        # Используемая процессом оперативная память в МБ
        return self.process.memory_info().rss / (1024 * 1024)
