# visualize_samples.py
import os
import cv2
import numpy as np
from dataset_yolo import ThermalPoseYOLODataset, NUM_KP
import matplotlib.pyplot as plt

def draw_keypoints_on_image_cv(img_bgr, kps_norm, color=(0,255,0)):
    h, w = img_bgr.shape[:2]
    for i in range(NUM_KP):
        x = kps_norm[2*i]   # normalized x
        y = kps_norm[2*i+1] # normalized y
        if x == 0 and y == 0:
            continue
        px = int(x * w)
        py = int(y * h)
        cv2.circle(img_bgr, (px, py), 3, color, -1)
    return img_bgr

if __name__ == '__main__':
    images_dir = r'C:\Users\SAKSHI\Desktop\MAJOR_PROJECT_GEU\dataset_openthermalpose\images'
    labels_dir = r'C:\Users\SAKSHI\Desktop\MAJOR_PROJECT_GEU\dataset_openthermalpose\labels'

    ds = ThermalPoseYOLODataset(images_dir, labels_dir, img_size=256)
    print("Dataset size:", len(ds))

    n = min(8, len(ds))
    os.makedirs('../output', exist_ok=True)

    for i in range(n):
        img_t, kp_flat, vis, bbox, img_name = ds[i]   # FIXED LINE

        img_np = (img_t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        out = draw_keypoints_on_image_cv(img_bgr, kp_flat)
        save_path = os.path.join('../output', f'viz_{i}_{os.path.basename(img_name)}')

        cv2.imwrite(save_path, out)
        print("Saved viz:", save_path)

    print("Done. Check the ../output folder.")
