import csv
import json
import random
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_CACHE_DIR = Path.home() / ".cache" / "coco_captions"

OUTPUT_BASE = Path("~/")

METADATA_COLUMNS = [
    'image_id', 'prompt_index', 'variant_index', 'prompt',
    'seed', 'variant_seed', 'num_inference_steps', 'guidance_scale',
    'model_name', 'timestamp',
]


def _ensure_coco_captions(split):
    COCO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    annotations_file = COCO_CACHE_DIR / f"captions_{split}.json"
    if annotations_file.exists():
        return annotations_file

    print(f"Downloading COCO {split} captions...")
    zip_path = COCO_CACHE_DIR / "annotations.zip"

    def _progress(block_num, block_size, total_size):
        pct = min(100, block_num * block_size * 100 // max(total_size, 1))
        print(f"\r  {pct}%", end='')

    urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, zip_path, reporthook=_progress)
    print()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            if 'captions_' in member and member.endswith('.json'):
                zf.extract(member, COCO_CACHE_DIR)
    sub = COCO_CACHE_DIR / "annotations"
    if sub.exists():
        for f in sub.glob("captions_*.json"):
            f.rename(COCO_CACHE_DIR / f.name)
        sub.rmdir()
    zip_path.unlink()
    return annotations_file


def prompts_select_all(num_samples=5000, seed=42, split='val2017'):
    with open(_ensure_coco_captions(split)) as f:
        data = json.load(f)

    captions = [ann['caption'] for ann in data['annotations']]
    rng = random.Random(seed)
    rng.shuffle(captions)
    captions = captions[:num_samples]
    print(f"Loaded {len(captions)} COCO prompts")
    return captions


def outdir_for(model, diversity):
    if diversity == "none":
        return OUTPUT_BASE / model / "none"
    return OUTPUT_BASE / model / f"eddy_{diversity}"


def _metadata_path(path, seed):
    return Path(path) / f"metadata_seed_{seed}.csv"


def outdir_init(path, seed):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    csv_path = _metadata_path(path, seed)
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(METADATA_COLUMNS)
    return path


def outdir_commit(path, seed, image, prompt_index, variant_index, prompt,
                  variant_seed, num_inference_steps, guidance_scale, model_name):
    path = Path(path)
    image_id = f"img_{prompt_index:05d}_v{variant_index}"
    image.save(path / f"{image_id}.png")

    with open(_metadata_path(path, seed), 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=METADATA_COLUMNS).writerow({
            'image_id': image_id,
            'prompt_index': prompt_index,
            'variant_index': variant_index,
            'prompt': prompt,
            'seed': seed,
            'variant_seed': variant_seed,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
        })
    return image_id
