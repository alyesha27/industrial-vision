"""Шаг 1. Объединение датасетов YOLO в один с единым списком классов.

Запуск:  python scripts/merge_datasets.py --config scripts/merge_config.yaml
Результат: datasets/unified/{images,labels}/{train,val,test} и datasets/unified/unified.yaml
Имена файлов получают префикс источника (ppe__..., dfire__...), чтобы не было коллизий.
"""
import argparse
import shutil
from pathlib import Path

import yaml

SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid", "validation"], "test": ["test"]}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split_dirs(root: Path, split: str):
    for alias in SPLIT_ALIASES[split]:
        for img_dir, lbl_dir in (
            (root / "images" / alias, root / "labels" / alias),
            (root / alias / "images", root / alias / "labels"),
        ):
            if img_dir.is_dir():
                return img_dir, lbl_dir
    return None, None


def source_names(root: Path):
    for candidate in (root / "data.yaml", root / "dataset.yaml", root / "data.yml"):
        if candidate.exists():
            names = yaml.safe_load(candidate.read_text(encoding="utf-8"))["names"]
            return names if isinstance(names, dict) else dict(enumerate(names))
    raise FileNotFoundError(f"Не найден data.yaml в {root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/merge_config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out = Path(cfg["output"])
    unified = {name: i for i, name in enumerate(cfg["unified_names"])}
    counts = {}

    for src in cfg["sources"]:
        root = Path(src["path"])
        names = source_names(root)
        # id в источнике -> id в едином списке (классы вне map отбрасываются)
        remap = {sid: unified[src["map"][n]] for sid, n in names.items() if n in src["map"]}
        print(f"[{src['name']}] классы источника {names} -> {remap}")

        for split in ("train", "val", "test"):
            img_dir, lbl_dir = find_split_dirs(root, split)
            if img_dir is None:
                print(f"  {split}: нет в источнике, пропуск")
                continue
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)

            n_img = 0
            for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
                lines = []
                lbl = lbl_dir / (img.stem + ".txt")
                if lbl.exists():
                    for line in lbl.read_text(encoding="utf-8").splitlines():
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        cid = int(float(parts[0]))
                        if cid in remap:
                            lines.append(" ".join([str(remap[cid])] + parts[1:]))
                new_stem = f"{src['name']}__{img.stem}"
                shutil.copy2(img, out / "images" / split / (new_stem + img.suffix.lower()))
                (out / "labels" / split / (new_stem + ".txt")).write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                n_img += 1
            counts[(src["name"], split)] = n_img
            print(f"  {split}: {n_img} изображений")

    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for n, i in unified.items()},
    }
    (out / "unified.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True), encoding="utf-8")
    print(f"\nГотово: {out / 'unified.yaml'}")
    print("Следующий шаг: python scripts/pseudo_label.py (достроить разметку учителями)")


if __name__ == "__main__":
    main()
