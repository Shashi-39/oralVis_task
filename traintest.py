import os
import shutil
import random


image_dir = "D:\oralVista\ToothNumber_TaskDataset\images"
label_dir = "D:\oralVista\ToothNumber_TaskDataset\labels"
output_root = "split_data"
train_ratio = 0.8
val_ratio = 0.1  # test = rest
seed = 42


random.seed(seed)
image_exts = (".jpg", ".jpeg", ".png")


for subset in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_root, "images", subset), exist_ok=True)
    os.makedirs(os.path.join(output_root, "labels", subset), exist_ok=True)


image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]
image_files = [f for f in image_files if os.path.exists(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"))]

random.shuffle(image_files)
n_total = len(image_files)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)

train_files = image_files[:n_train]
val_files = image_files[n_train:n_train + n_val]
test_files = image_files[n_train + n_val:]

splits = [("train", train_files), ("val", val_files), ("test", test_files)]


for split_name, files in splits:
    for fname in files:
        stem = os.path.splitext(fname)[0]
        # copy image
        shutil.copy2(os.path.join(image_dir, fname), os.path.join(output_root, "images", split_name, fname))
        # copy label
        shutil.copy2(os.path.join(label_dir, stem + ".txt"), os.path.join(output_root, "labels", split_name, stem + ".txt"))

 
print(f"  Train: {len(train_files)}")
print(f"  Val:   {len(val_files)}")
print(f"  Test:  {len(test_files)}")
print(f"Saved in '{output_root}/images/' and '{output_root}/labels/'")
