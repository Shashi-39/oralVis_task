import os
import cv2

# Folders (adjust if needed)
image_dir = "D:\oralVista\ToothNumber_TaskDataset\images"
label_dir = "D:\oralVista\ToothNumber_TaskDataset\labels"
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# For each image in image_dir
for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
    out_path = os.path.join(output_dir, filename)

    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not read {img_path}")
        continue

    H, W = image.shape[:2]

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                # Convert from YOLO to pixel coords
                x1 = int((cx - w / 2) * W)
                y1 = int((cy - h / 2) * H)
                x2 = int((cx + w / 2) * W)
                y2 = int((cy + h / 2) * H)

                # Draw box and label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(int(cls)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite(out_path, image)

print("✅ Done. Annotated images saved to 'output/'")
