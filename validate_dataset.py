from pathlib import Path
import yaml

ROOT = Path("dataset")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_CLASSES = 4

print("\n===== APPLE DATASET VALIDATION =====")

if not ROOT.exists():
    raise SystemExit("ERROR: dataset folder not found!")

config_file = ROOT / "data.yaml"

if not config_file.exists():
    raise SystemExit("ERROR: data.yaml not found!")

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

print("\nClasses:", config.get("names"))
print("Number of classes:", config.get("nc"))

if config.get("nc") != NUM_CLASSES:
    print("WARNING: Expected 4 classes.")

if len(config.get("names", [])) != NUM_CLASSES:
    print("ERROR: Class names count does not match nc.")

total_images = 0
total_labels = 0
total_boxes = 0
errors = []

for split in SPLITS:
    image_dir = ROOT / split / "images"
    label_dir = ROOT / split / "labels"

    print(f"\n--- {split.upper()} ---")

    if not image_dir.exists() or not label_dir.exists():
        errors.append(f"{split}: images or labels folder missing")
        print("ERROR: Missing folders")
        continue

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without labels:", len(missing_labels))
    print("Labels without images:", len(orphan_labels))

    total_images += len(images)
    total_labels += len(labels)

    if missing_labels:
        errors.append(f"{split}: images missing labels")

    if orphan_labels:
        errors.append(f"{split}: labels without images")

    for label_file in labels:
        with open(label_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_file.name}, line {line_no}: invalid format"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label_file.name}, line {line_no}: non-numeric value"
                )
                continue

            if not 0 <= class_id < NUM_CLASSES:
                errors.append(
                    f"{label_file.name}: invalid class ID {class_id}"
                )
                continue

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                errors.append(
                    f"{label_file.name}: coordinates outside 0-1"
                )
                continue

            if w <= 0 or h <= 0:
                errors.append(
                    f"{label_file.name}: invalid box dimensions"
                )
                continue

            total_boxes += 1

    print("Split checked.")

print("\n===== FINAL SUMMARY =====")
print("Total images:", total_images)
print("Total label files:", total_labels)
print("Total bounding boxes:", total_boxes)

if errors:
    print("\nISSUES FOUND:")
    for error in errors[:30]:
        print("-", error)
    print("Total issues:", len(errors))
else:
    print("\nSUCCESS: No annotation or folder errors found!")
