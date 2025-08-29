from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # or yolov8n.pt for lighter/faster

    # IMPORTANT on Windows: keep workers small (0–2). 0 is safest.
    model.train(
        data="split_data/data.yaml",
        epochs=10,
        imgsz=640,
        batch=16,          # reduce if OOM (try 8/4)
        device="cuda:0",
        workers=0,         # <- key change for Windows
        deterministic=True # stable runs
    )

    # Validate (gets P/R/mAP and saves confusion matrix)
    model.val(split="val", imgsz=640, device="cuda:0", workers=0)

    # Predict a few test images and save annotated copies
    model.predict(source="split_data/images/test", imgsz=640, device="cuda:0", save=True, workers=0)

if __name__ == "__main__":   # <- key line for Windows spawn
    main()
