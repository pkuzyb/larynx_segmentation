import os
import json
import torch
import numpy as np
import cv2
import copy
from torch.utils.data import Dataset
from detectron2.structures import BoxMode
from pycocotools import mask as mask_util
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


# -----------------------
# Helper functions
# -----------------------
def connect_and_close_contours(data, lower_label='vocal_folds', upper_label='ventricular_folds', ventricle_label='ventricle'):
    """Connect vocal/ventricular folds with ventricle points to close contours"""
    ventricle_points = None
    for shape in data.get("shapes", []):
        if shape.get("label") == ventricle_label:
            ventricle_points = np.array(shape["points"])
            break
    if ventricle_points is None:
        return
    
    leftmost_idx = np.argmin(ventricle_points[:, 0])
    rightmost_idx = np.argmax(ventricle_points[:, 0])
    lower_ventricle_points = ventricle_points[leftmost_idx:rightmost_idx+1][::-1]
    upper_ventricle_points = np.vstack([ventricle_points[rightmost_idx:], ventricle_points[leftmost_idx, :]])

    for shape in data.get("shapes", []):
        label = shape.get("label")
        points = np.array(shape["points"])
        if label == lower_label:
            new_points = np.vstack([points, lower_ventricle_points])
            shape["points"] = new_points.tolist()
        elif label == upper_label:
            new_points = np.vstack([upper_ventricle_points, points])
            shape["points"] = new_points.tolist()

# -----------------------
# Dataset class
# -----------------------

class Mask2FormerDataset(Dataset):
    LABEL_MAP = {
        'thyroid_cartilage': 0, 'ventricle': 1, 'vocal_folds': 2,
        'arytenoid_cartilage': 3, 'epiglottis': 4, 'ventricular_folds': 5,
    }

    def __init__(self, json_folder, label_map=None, resize_to=1024):
        self.json_folder = json_folder
        self.json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
        self.resize_to = resize_to
        self.label_map = label_map or self.LABEL_MAP

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        json_path = os.path.join(self.json_folder, json_file)

        with open(json_path, 'r') as f:
            data = json.load(f)

        img_name = data['imagePath']
        img_path = os.path.join(self.json_folder, img_name)
        
        # Get original dims to calculate scale
        tmp_img = cv2.imread(img_path)
        orig_h, orig_w = tmp_img.shape[:2]

        # rescale the annotations here
        if self.resize_to:
            scale_x = self.resize_to / orig_w
            scale_y = self.resize_to / orig_h
            h, w = self.resize_to, self.resize_to
        else:
            scale_x = scale_y = 1.0
            h, w = orig_h, orig_w

        data_copy = copy.deepcopy(data)
        connect_and_close_contours(data_copy)

        annotations = []
        for shape in data_copy['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.float32)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points = np.vstack([points, points[0]])
            points_int = np.clip(np.round(points).astype(np.int32), 0, [w-1, h-1])

            # Rasterize smooth mask at high-res
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points_int], 1)

            rle = mask_util.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode("utf-8")
            x, y, bw, bh = cv2.boundingRect(points_int)

            annotations.append({
                "bbox": [x, y, bw, bh],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": rle,
                "category_id": self.label_map.get(label, 0),
            })

        # Return metadata only. The actual images are not rescaled here. The MAPPER will load the actual pixels.
        return {
            "file_name": img_path,
            "image_id": idx,
            "height": h,
            "width": w,
            "annotations": annotations
        }
    
class UnlabeledMask2FormerDataset(Dataset):
    def __init__(self, img_folder, resize_to=1024):
        self.img_folder = img_folder
        self.img_files = [f for f in os.listdir(img_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.resize_to = resize_to

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        
        # Only get dimensions
        img_info = cv2.imread(img_path)
        if img_info is None: return None
        orig_h, orig_w = img_info.shape[:2]

        scale = self.resize_to / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)

        return {
            "file_name": img_path,
            "image_id": f"unlabeled_{idx}",
            "height": new_h,
            "width": new_w,
            "annotations": []
        }
    
# -----------------------
# Test
# -----------------------
if __name__ == "__main__":
    labeled_json_folder = './lary_seg/data/train/mand_01'
    unlabeled_img_folder = './lary_seg/data/unlabelled'
    output_folder = './lary_seg/debug_load'
    os.makedirs(output_folder, exist_ok=True)

    # 1. Initialize both datasets
    labeled_ds = Mask2FormerDataset(labeled_json_folder, resize_to=1024)
    unlabeled_ds = UnlabeledMask2FormerDataset(unlabeled_img_folder, resize_to=1024)

    print(f"Labeled samples found: {len(labeled_ds)}")
    print(f"Unlabeled samples found: {len(unlabeled_ds)}")
