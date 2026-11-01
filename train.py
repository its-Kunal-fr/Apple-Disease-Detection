
from ultralytics import YOLO
import torch

def main():

    print("================================")
    print(" APPLE DISEASE DETECTION")
    print(" YOLO11 MODEL TRAINING")
    print("================================")

    # Check GPU
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("CUDA unavailable. Using CPU.")
        device = "cpu"

    # Load pretrained YOLO11 Nano model
    model = YOLO("yolo11n.pt")

    print("\nStarting training...\n")

    results = model.train(
        data="dataset/data.yaml",
        epochs=60,
        imgsz=640,
        batch=8,
        device=device,
        workers=2,
        patience=15,
        project="runs",
        name="apple_disease",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        plots=True,
        save=True,
        seed=42
    )

    print("\nTraining completed!")
    print("Best model:")
    print("runs/apple_disease/weights/best.pt")

    print("\nTraining results:", results)

if __name__ == "__main__":
    main()
