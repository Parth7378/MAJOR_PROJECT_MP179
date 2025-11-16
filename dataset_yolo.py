# dataset_yolo.py
# FINAL VERSION – Correct for OpenThermalPose dataset
# Supports: multiple persons concatenated on ONE LINE in each .txt file
# Selects the MAIN person = the one with LARGEST bounding box (closest to camera)

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

NUM_KP = 17
PERSON_SIZE = 1 + 4 + NUM_KP * 3    # 1 cls + 4 box + 51 keypoints = 56 numbers per person


class ThermalPoseYOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=256):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size

        # Collect valid image-label pairs
        imgs = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.items = []
        for img_name in imgs:
            base = os.path.splitext(img_name)[0]
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, base + ".txt")

            if os.path.isfile(img_path) and os.path.isfile(label_path):
                self.items.append({"img": img_path, "label": label_path})

        print(f"[INFO] Loaded {len(self.items)} matched image-label pairs.")

    def __len__(self):
        return len(self.items)

    # ----------------------------------------------------------
    # Parse label file (One line contains ALL persons sequentially)
    # ----------------------------------------------------------
    def _parse_label_file(self, label_path):
        with open(label_path, "r") as f:
            txt = f.read().strip()

        if txt == "":
            return None, None, None

        nums = [float(x) for x in txt.split()]
        total = len(nums)

        if total < PERSON_SIZE:
            return None, None, None

        persons = total // PERSON_SIZE  # number of persons

        best_bbox = None
        best_kps = None
        best_vis = None
        best_area = -1

        for p in range(persons):
            start = p * PERSON_SIZE
            end = start + PERSON_SIZE
            data = nums[start:end]

            cls_id = int(data[0])
            x_c, y_c, w, h = data[1:5]    # YOLO-format normalized bbox

            area = w * h    # largest box → closest person

            kp_nums = data[5:]  # 51 values for 17 joints

            kps = np.zeros((NUM_KP, 2), dtype=np.float32)
            vis = np.zeros((NUM_KP,), dtype=np.float32)

            for i in range(NUM_KP):
                xi = kp_nums[3*i]
                yi = kp_nums[3*i + 1]
                vi = int(kp_nums[3*i + 2])

                kps[i] = [xi, yi]
                vis[i] = vi

            # SELECT PERSON WITH LARGEST AREA (closest to camera)
            if area > best_area:
                best_area = area
                best_bbox = np.array([x_c, y_c, w, h], dtype=np.float32)
                best_kps = kps
                best_vis = vis

        return best_bbox, best_kps, best_vis

    # ----------------------------------------------------------
    # Load image and prepare tensors
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        rec = self.items[idx]

        # Load & resize image
        img = Image.open(rec["img"]).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img_t = TF.to_tensor(img)

        # Parse keypoints
        bbox, kps, vis = self._parse_label_file(rec["label"])

        if bbox is None:
            bbox = np.zeros((4,), dtype=np.float32)
            kps = np.zeros((NUM_KP, 2), dtype=np.float32)
            vis = np.zeros((NUM_KP,), dtype=np.float32)

        kp_flat = kps.flatten().astype(np.float32)

        return img_t, kp_flat, vis.astype(np.float32), bbox.astype(np.float32), rec["img"]
