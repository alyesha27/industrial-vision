"""Меню выбора источника видео. Запуск: python launcher.py

Без меню:
  python launcher.py --source 0
  python launcher.py --source data/test.mp4
  SOURCE=/dev/video0 python launcher.py        (удобно для Docker)
"""
import os
import sys
import subprocess
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import cv2

DATA_DIRS = [Path("data")]                        # папки с тестовыми видео
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_CAMERAS = 4


def find_cameras():
    found = []
    for index in range(MAX_CAMERAS):
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(index)
        finally:
            cap.release()
    return found


def find_videos():
    files = []
    for folder in DATA_DIRS:
        if folder.exists():
            files += [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT]
    return sorted(files)


def build_menu(cameras):
    items = []
    for number, index in enumerate(cameras, 1):
        items.append((f"Камера {number} (индекс {index})", index))
    for path in find_videos():
        items.append((f"Файл: {path}", str(path)))
    items.append(("RTSP-адрес (ввести вручную)", "@rtsp"))
    items.append(("Путь к файлу (ввести вручную)", "@path"))
    return items


def needs_headless():
    # на Linux без графической сессии (плата по SSH, Docker) окно открыть нельзя
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY") \
        and not os.environ.get("WAYLAND_DISPLAY")


def run(source):
    cmd = [sys.executable, "-m", "src.inference.detect_camera", "--source", str(source)]
    if needs_headless():
        cmd.append("--no-show")
    print(f"\nЗапуск: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nОстановлено")


def main():
    if "--source" in sys.argv:
        return run(sys.argv[sys.argv.index("--source") + 1])
    if os.environ.get("SOURCE"):
        return run(os.environ["SOURCE"])

    print("Поиск камер...")
    cameras = find_cameras()
    while True:
        items = build_menu(cameras)
        print("\n=== Выбор источника видео ===")
        for number, (title, _) in enumerate(items, 1):
            print(f"{number}. {title}")
        print("r. Обновить список камер")
        print("0. Выход")
        choice = input("Ваш выбор: ").strip().lower()

        if choice == "0":
            return
        if choice == "r":
            print("Поиск камер...")
            cameras = find_cameras()
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(items):
            print("Неверный выбор")
            continue

        source = items[int(choice) - 1][1]
        if source == "@rtsp":
            source = input("RTSP-адрес (rtsp://...): ").strip()
            if not source.startswith("rtsp://"):
                print("Адрес должен начинаться с rtsp://")
                continue
        elif source == "@path":
            source = input("Путь к видеофайлу: ").strip().strip('"')
            if not Path(source).exists():
                print("Файл не найден")
                continue
        run(source)


if __name__ == "__main__":
    main()
