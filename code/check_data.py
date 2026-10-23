from pathlib import Path
from datasets import load_dataset

repository_dir = Path(__file__).resolve().parents[1]
data_dir = repository_dir.parent / "wikitext-103-raw-v1"

data_files = {
    "train": [
        str(data_dir / "train-00000-of-00002.parquet"),
        str(data_dir / "train-00001-of-00002.parquet"),
    ],
    "validation": [
        str(data_dir / "validation-00000-of-00001.parquet")
    ],
    "test": [
        str(data_dir / "test-00000-of-00001.parquet")
    ],
}

for files in data_files.values():
    for filename in files:
        if not Path(filename).is_file():
            raise FileNotFoundError(f"Missing dataset file: {filename}")

dataset = load_dataset("parquet", data_files=data_files)

print("\nDataset loaded successfully!")
print(dataset)

print("\nNumber of rows per split:")
for split, data in dataset.items():
    print(f"{split}: {len(data):,}")

print("\nColumns:", dataset["train"].column_names)

print("\nFirst three non-empty training rows:")
shown = 0
for row in dataset["train"]:
    text = row["text"]
    if text.strip():
        print(f"\n--- Sample {shown + 1} ---")
        print(text[:500])
        shown += 1
        if shown == 3:
            break
