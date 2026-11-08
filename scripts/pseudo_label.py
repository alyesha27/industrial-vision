"""Шаг 2. Достройка разметки моделями-учителями.

Проблема объединения датасетов: в D-Fire нет разметки людей и касок, в PPE нет
огня и т.д. Если оставить как есть, обучение будет считать людей на кадрах с огнём
фоном. Поэтому для каждого изображения учителя дописывают только те классы,
которых нет в исходном датасете этого изображения.

Запуск:  python scripts/pseudo_label.py --config scripts/merge_config.yaml
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/merge_config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out = Path(cfg["output"])
    unified = {name: i for i, name in enumerate(cfg["unified_names"])}
    conf = cfg.get("teacher_conf", 0.5)
    # какие классы размечены в каждом источнике (их учителя не трогают)
    labeled = {s["name"]: {unified[v] for v in s["map"].values()} for s in cfg["sources"]}

    teachers = [(YOLO(t["model"]), t["classes"]) for t in cfg["teachers"]]
    added = {name: 0 for name in unified}

    for split in ("train", "val", "test"):
        img_dir = out / "images" / split
        if not img_dir.is_dir():
            continue
        images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        print(f"{split}: {len(images)} изображений")

        for index, img in enumerate(images, 1):
            source_name = img.name.split("__", 1)[0]
            skip = labeled.get(source_name, set())
            new_lines = []
            for model, class_map in teachers:
                result = model.predict(str(img), conf=conf, verbose=False)[0]
                for box in result.boxes:
                    teacher_name = model.names[int(box.cls.item())]
                    if teacher_name not in class_map:
                        continue
                    cid = unified[class_map[teacher_name]]
                    if cid in skip:
                        continue
                    cx, cy, bw, bh = box.xywhn[0].tolist()
                    new_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    added[class_map[teacher_name]] += 1
            if new_lines:
                lbl = out / "labels" / split / (img.stem + ".txt")
                with lbl.open("a", encoding="utf-8") as f:
                    f.write("\n".join(new_lines) + "\n")
            if index % 500 == 0:
                print(f"  {index}/{len(images)}")

    print("\nДобавлено рамок по классам:", {k: v for k, v in added.items() if v})
    print("Обучение:  yolo detect train model=yolo26n.pt "
          f"data={out / 'unified.yaml'} epochs=50 imgsz=640 batch=16")


if __name__ == "__main__":
    main()
