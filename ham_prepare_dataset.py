"""
Prepare HAM10000 into train/test folder structure for MorphNN.
Splits by lesion_id (not image) to prevent data leakage from duplicate lesion images.
80/20 stratified split on lesion level.
Creates symlinks in Ham Dataset/train/<class>/ and Ham Dataset/test/<class>/
"""
import os, shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT     = Path(__file__).resolve().parent
HAM_DIR  = ROOT / "Ham Dataset"
META_CSV = HAM_DIR / "HAM10000_metadata.csv"
PART1    = HAM_DIR / "HAM10000_images_part_1"
PART2    = HAM_DIR / "HAM10000_images_part_2"
SEED     = 42

df = pd.read_csv(META_CSV)
print(f"Total images: {len(df)}")
print(f"Classes: {sorted(df['dx'].unique())}")
print(f"Class counts:\n{df['dx'].value_counts().to_string()}\n")

# Build image path lookup
img_lookup = {}
for part in [PART1, PART2]:
    for f in part.iterdir():
        img_lookup[f.stem] = f   # ISIC_XXXXXXX → Path

# Verify all images exist
missing = [r['image_id'] for _, r in df.iterrows() if r['image_id'] not in img_lookup]
print(f"Missing images: {len(missing)}")

# Deduplicate by lesion_id — one record per lesion for the split
lesion_df = df.drop_duplicates('lesion_id')[['lesion_id','dx']].reset_index(drop=True)
print(f"Unique lesions: {len(lesion_df)}")

# Stratified 80/20 split on lesion_id level
train_lesions, test_lesions = train_test_split(
    lesion_df['lesion_id'].to_numpy(),
    test_size=0.2,
    stratify=lesion_df['dx'].to_numpy(),
    random_state=SEED
)
train_set = set(train_lesions)
test_set  = set(test_lesions)
print(f"Train lesions: {len(train_set)}, Test lesions: {len(test_set)}")

# Create folder structure and symlinks
for split in ['train', 'test']:
    split_dir = HAM_DIR / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    for cls in sorted(df['dx'].unique()):
        (split_dir / cls).mkdir(parents=True, exist_ok=True)

train_counts = {}; test_counts = {}
for _, row in df.iterrows():
    img_id = row['image_id']
    cls    = row['dx']
    lesion = row['lesion_id']
    if img_id not in img_lookup:
        continue
    src = img_lookup[img_id]
    if lesion in train_set:
        dst = HAM_DIR / 'train' / cls / f"{img_id}.jpg"
        train_counts[cls] = train_counts.get(cls, 0) + 1
    else:
        dst = HAM_DIR / 'test' / cls / f"{img_id}.jpg"
        test_counts[cls] = test_counts.get(cls, 0) + 1
    if not dst.exists():
        os.symlink(src, dst)

print("\nTrain images per class:")
for cls in sorted(train_counts): print(f"  {cls}: {train_counts[cls]}")
print(f"  TOTAL: {sum(train_counts.values())}")
print("\nTest images per class:")
for cls in sorted(test_counts): print(f"  {cls}: {test_counts[cls]}")
print(f"  TOTAL: {sum(test_counts.values())}")
print("\nDataset ready.")
