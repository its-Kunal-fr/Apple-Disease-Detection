
from ultralytics import YOLO
from pathlib import Path

model = YOLO(
    r"runs/detect/runs/apple_disease/weights/best.pt"
)

# Select an image from the test dataset
test_folder = Path("dataset/test/images")

images = [
    p for p in test_folder.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]

if not images:
    raise SystemExit("No test images found!")

image_path = str(images[0])

print("Testing image:", image_path)

results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    device=0,
    save=True
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Disease: {model.names[class_id]} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

print("\nPrediction saved successfully!")
