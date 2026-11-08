# Industrial Vision

Программный модуль технического зрения для промышленного мобильного роботизированного комплекса.

## Возможности

- Получение видеопотока с камеры, из видеофайла или по RTSP, переподключение при потере потока.
- Обнаружение и отслеживание людей и транспорта (Track ID).
- Контроль нахождения людей и транспорта в запрещённой зоне (привязка по нижнему центру, центру или обоим).
- Контроль средств индивидуальной защиты: касок и жилетов.
- Обнаружение огня, дыма и воды/разлива.
- Регистрация событий с подтверждающими снимками.
- REST API для получения состояния системы и событий.
- Журналирование и метрики производительности.
- Меню выбора источника видео при запуске.
- Запуск в Docker, в том числе на одноплатном компьютере (arm64).

## Технологии

Python 3.13, OpenCV, NumPy, Ultralytics YOLO, FastAPI, Uvicorn, Pytest, ONNX Runtime, Docker.

## Структура проекта

- `launcher.py` — меню выбора источника видео и запуск детектора.
- `src/capture/` — получение видеопотока.
- `src/inference/` — детектор (`detector.py`) и главный цикл (`detect_camera.py`).
- `src/analytics/` — события по объектам, СИЗ, эвристика разлива.
- `src/zones/` — геометрия зон и режимы привязки к зоне (`anchors.py`).
- `src/events/` — Event Engine, сохранение событий.
- `src/api/` — REST API.
- `src/monitoring/` — логирование и метрики.
- `scripts/` — benchmark, загрузка моделей, подготовка единой модели (`merge_datasets.py`, `pseudo_label.py`).
- `configs/` — конфигурация (`configuration.yaml`).
- `models/` — веса моделей (в репозиторий не входят).
- `data/` — тестовые видео (в репозиторий не входят).
- `events/` — сохранённые события и снимки.
- `tests/` — автоматические тесты.

## 1. Подготовка окружения

Нужен Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS: `source .venv/bin/activate`.

Файлы моделей должны лежать по путям из `configs/configuration.yaml` (папка `models/`).

## 2. Запуск через меню

```powershell
python launcher.py
```

Меню показывает найденные камеры (Камера 1, Камера 2 ...), видеофайлы из папки `data/` (включая вложенные),
пункты «RTSP-адрес» и «Путь к файлу» для ручного ввода. После остановки детектора меню появляется снова.
`r` обновляет список камер, `0` — выход.

Запуск без меню:

```powershell
python launcher.py --source 0
python launcher.py --source "data\test_video.mp4"
```

Напрямую, без лаунчера:

```powershell
python -m src.inference.detect_camera --source 0
python -m src.inference.detect_camera --source "data\test_video.mp4" --no-show
```

`--no-show` отключает окно с видео (для серверов, Docker, платы без экрана). Выход из окна: клавиша `Q`.

## 3. Запуск REST API

В отдельном терминале с активным окружением:

```powershell
python -m uvicorn src.api.server:app --reload
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- `GET /health`, `GET /status`, `GET /events`, `GET /events/{id}`, `GET /cameras`
- `POST /config/thresholds`, `POST /config/zones`

API только читает данные с диска, обработка видео запускается отдельно.

## 4. Запуск в Docker (ПК и одноплатный компьютер)

Образ собирается для amd64 и arm64. Состав: сервис `detector` (обработка без окна) и сервис `api` (порт 8000), общий каталог `events`.

1. Положите веса в `models/`, тестовые видео в `data/`.
2. Скопируйте `.env.example` в `.env` и задайте источник:
   - камера: `SOURCE=/dev/video0`
   - файл: `SOURCE=data/test.mp4` (и закомментируйте `devices` в `docker-compose.yml`)
   - RTSP: `SOURCE=rtsp://user:pass@192.168.1.10:554/stream`
3. Запуск:

```bash
docker compose up -d --build
curl http://localhost:8000/health
docker compose logs -f detector
```

Остановка: `docker compose down`.

Сборка образа для платы на обычном ПК:

```bash
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx build --platform linux/arm64 -t abd-vision:1.0 --load .
```

Замечания:
- Доступ к камере из контейнера работает на Linux (`/dev/video*`). На Windows/macOS для проверки используйте видеофайл.
- После изменения `configs/configuration.yaml` (в том числе через API) перезапустите детектор: `docker compose restart detector`.
- Для ускорения на CPU платы можно экспортировать модель в ONNX: `yolo export model=models/unified.pt format=onnx`.

## 5. Конфигурация

Основные параметры в `configs/configuration.yaml`.

**Источник и модели.** Если существует файл `models/unified.pt`, работает одна модель на все классы
(`person, car, truck, bus, helmet, head, vest, fire, smoke, water`), пороги задаются в `models.class_confidence`.
Если файла нет, используется режим из трёх моделей (`detector_path`, `ppe_path`, `fire_smoke_path`).

**Привязка к зоне.** Параметр `zones.anchor_mode` задаёт отдельно для людей и транспорта:

| Режим | Что проверяется |
|---|---|
| `bottom_center` | нижняя центральная точка рамки (точка контакта с полом) |
| `center` | центр рамки |
| `any` | любая из двух точек |

Для людей по умолчанию `bottom_center` (центр рамки может оказаться в зоне, когда человек стоит за её границей),
для транспорта `any`.

**События.** `events.confirmation_n` из `events.confirmation_m` кадров и пауза `events.cooldown_sec`.

## 6. Единая модель YOLO

Одна модель вместо трёх. Подготовка:

```bash
python scripts/merge_datasets.py --config scripts/merge_config.yaml   # объединение датасетов
python scripts/pseudo_label.py   --config scripts/merge_config.yaml   # достройка разметки текущими моделями
yolo detect train model=yolo26n.pt data=datasets/unified/unified.yaml epochs=50 imgsz=640 batch=16
```

Датасеты, пути и соответствие классов указываются в `scripts/merge_config.yaml`. Результат обучения (`best.pt`)
сохраните как `models/unified.pt`.

## 7. Тесты

```powershell
python -m pytest -q
```

## 8. Результаты работы

События сохраняются в `events/ГГГГ/ММ/ДД/<uuid>/` (файлы `event.json` и `snapshot.jpg`).
Для демонстрации: видеопоток с рамками и зоной, не менее двух разных событий, файлы `event.json` и снимок, работа API.

## 9. Ограничения

Результаты зависят от качества видео, освещения, ракурса камеры, моделей и порогов уверенности.
Если единая модель не содержит класс `water`, разлив определяется эвристикой, а не нейросетью.
RTSP поддерживается OpenCV, но на реальной IP-камере не проверялся. MQTT не реализован.
Система предназначена для демонстрации и проверки сценариев и не является гарантированным средством
обеспечения промышленной безопасности.
