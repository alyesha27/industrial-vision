import cv2
import time
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Ошибка: не удалось открыть камеру")
else:
    print("Камера запущена. Нажмите Q для выхода.")

    frames = 0
    start_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Не удалось получить кадр")
            break

        frames += 1
        elapsed = time.time() - start_time
        fps = frames / elapsed if elapsed > 0 else 0

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

    camera.release()
    cv2.destroyAllWindows()
