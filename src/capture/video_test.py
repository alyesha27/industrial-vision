
import cv2

from video_capture import VideoSource, FPSMeter


# 0 — встроенная или подключённая камера
source = VideoSource(source=0, reconnect_sec=2)
fps_meter = FPSMeter()

try:
    source.open()
    print("Камера подключена. Нажмите Q для выхода.")

    while True:
        frame = source.read()

        # Рассчитываем FPS
        fps = fps_meter.tick()

        # Выводим FPS на изображение
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Industrial Vision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as error:
    print(f"Ошибка видеопотока: {error}")

finally:
    source.close()
    cv2.destroyAllWindows()
