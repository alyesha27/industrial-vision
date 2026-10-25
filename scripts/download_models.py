import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

Path("models").mkdir(exist_ok=True)
src = hf_hub_download(repo_id="rabahdev/fire-smoke-yolov8n", filename="best.pt")
shutil.copy(src, "models/fire_smoke.pt")
print("Готово: models/fire_smoke.pt")