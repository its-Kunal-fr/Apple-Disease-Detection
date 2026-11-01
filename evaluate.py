
from ultralytics import YOLO
import torch
from multiprocessing import freeze_support

def main():

    print("\n===== APPLE DISEASE TEST EVALUATION =====")

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(
        r"runs/detect/runs/apple_disease/weights/best.pt"
    )

    print("\nEvaluating on unseen test images...")

    metrics = model.val(
        data="dataset/data.yaml",
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project="runs",
        name="apple_test",
        exist_ok=True
    )

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    print("\nTest evaluation completed successfully!")

if __name__ == "__main__":
    freeze_support()
    main()
