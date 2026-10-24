from pathlib import Path
import yaml


def load_config(path="configs/configuration.yaml"):
    # Читаем YAML в UTF-8 (важно для Windows)
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)