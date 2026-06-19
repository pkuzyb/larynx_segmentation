import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util
import copy
import random
from tabulate import tabulate # For a clean results table
import torch.nn.functional as F
import shutil
import sys
from medpy import metric

from dataset import Mask2FormerDataset  # your dataset.py

from detectron2.config import get_cfg
from mask2former import add_maskformer2_config
from detectron2.projects.deeplab import add_deeplab_config
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.modeling import build_model
from detectron2.engine import DefaultTrainer, hooks

torch.cuda.empty_cache()
sys.path.append("./Mask2Former")

# -----------------------------
# Dataset Registration
# -----------------------------

# --- dataset creation functions ---
def create_reduced_dataset(original_folder, target_folder, reduction_ratio, seed=42, record_json="selected_pairs.json"):
    """
    Create a reduced dataset for each subject independently:
    - Copy selected image + JSON pairs into a flat target folder
    - Save a JSON file that tracks which pairs were selected
    
    original_folder: contains one folder per subject
    target_folder: where selected images/JSONs will be copied
    reduction_ratio: fraction of images per subject to keep
    seed: random seed
    record_json: path to save the tracking JSON file
    """
    rng = random.Random(seed)

    # clean target folder
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder, exist_ok=True)

    total_selected = 0
    all_selected = {}  # for tracking

    for subject in sorted(os.listdir(original_folder)):
        subject_path = os.path.join(original_folder, subject)
        if not os.path.isdir(subject_path):
            continue

        # collect image + JSON pairs
        pairs = {}
        for f in sorted(os.listdir(subject_path)):
            stem, ext = os.path.splitext(f)
            ext = ext.lower()
            if ext not in [".jpg", ".png", ".json"]:
                continue
            pairs.setdefault(stem, {})[ext] = os.path.join(subject_path, f)

        valid = [pairs[k] for k in sorted(pairs) if ".jpg" in pairs[k] and ".json" in pairs[k]]
        if len(valid) == 0:
            continue

        # sample per-subject
        n_select = max(1, int(len(valid) * reduction_ratio))
        sampled = rng.sample(valid, n_select)

        # copy selected files to target folder
        for pair in sampled:
            for src in pair.values():
                dst = os.path.join(target_folder, os.path.basename(src))
                shutil.copy2(src, dst)

        # record selected filenames for tracking
        all_selected[subject] = [
            {ext: os.path.basename(path) for ext, path in pair.items()} for pair in sampled
        ]

        print(f"{subject}: selected {n_select}/{len(valid)}")
        total_selected += n_select

    # save tracking JSON
    with open(record_json, "w", encoding="utf-8") as f:
        json.dump(all_selected, f, indent=2)

    print(f"\nTOTAL selected pairs: {total_selected}")
    print(f"Tracking JSON saved to: {record_json}")

def get_lary_dicts(json_folder):
    ## the annotations are rescaled to 1024 using the dataset function
    dataset = Mask2FormerDataset(json_folder,resize_to=1024)
    records = []
    for record in dataset:
        records.append(record)
    return records

# -----------------------------
# Data Augmentation Mapper
# -----------------------------


def mapper(dataset_dict, is_strong=False):
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # Load image 
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    
    # The images are rescaled to 1024 here
    h, w = dataset_dict["height"], dataset_dict["width"]
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert to tensor
    dataset_dict["image"] = torch.from_numpy(image.transpose(2, 0, 1).astype("float32"))

    # Process Instances
    annos = dataset_dict.pop("annotations", [])
    masks, boxes, classes = [], [], []

    for ann in annos:
        mask = mask_util.decode(ann["segmentation"]).astype(np.uint8)
        x, y, bw, bh = ann["bbox"]
        masks.append(torch.from_numpy(mask))
        boxes.append(torch.tensor([x, y, x + bw, y + bh], dtype=torch.float32))
        classes.append(ann["category_id"])

    instances = Instances((h, w))
    if len(masks) > 0:
        instances.gt_boxes = Boxes(torch.stack(boxes))
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_masks = torch.stack(masks)
    else:
        instances.gt_boxes = Boxes(torch.empty((0, 4)))
        instances.gt_classes = torch.empty((0,), dtype=torch.int64)
        instances.gt_masks = torch.empty((0, h, w), dtype=torch.uint8)

    dataset_dict["instances"] = instances
    return dataset_dict

# -----------------------------
# Trainer with Early Stopping
# -----------------------------
#--- Custom early stopping exception ---
class ValidationLossHook(hooks.HookBase):
    """
    Hook to compute validation loss and do early stopping.
    """
    def __init__(self, model, val_loader, patience=10, checkpointer=None, results_folder="./", print_freq=100):
        self.model = model
        self.val_loader = val_loader
        self.patience = patience
        self.checkpointer = checkpointer
        self.results_folder = results_folder
        self.print_freq = print_freq
        self.best_val_loss = float("inf")
        self.no_improve_count = 0

    def after_step(self):
        if self.trainer.iter % self.print_freq != 0:
            return

        val_losses = []

        # Temporarily keep model in train mode to compute losses
        self.model.train()
        with torch.no_grad():  
            for batch in self.val_loader:
                loss_dict = self.model(batch)

                # loss_dict should be a dict of tensors
                if not isinstance(loss_dict, dict):
                    raise TypeError(f"Expected dict of losses, got {type(loss_dict)}")
                total_loss = sum(loss_dict.values())
                val_losses.append(total_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        storage = get_event_storage()
        storage.put_scalar("val_loss", avg_val_loss)
        print(f"[Iter {self.trainer.iter}] Validation loss: {avg_val_loss:.4f}")

        # --- Early stopping ---
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.no_improve_count = 0
            # Save best model
            if self.checkpointer:
                self.checkpointer.save("best_model")
        else:
            self.no_improve_count += 1
        
        if self.no_improve_count >= self.patience:
            print(f"Early stopping triggered at iteration {self.trainer.iter}.")
            if self.checkpointer:
                self.checkpointer.save("final_model")
            raise StopTrainingException()
        

class StopTrainingException(Exception):
    """Custom exception to stop training cleanly"""
    pass


class TrainerWithEarlyStopping(DefaultTrainer):
    """
    Trainer that computes validation loss and stops early.
    """
    def __init__(self, cfg, patience=10, results_folder="./"):
        self.patience = patience
        self.results_folder = results_folder

        # --- Build validation loader internally ---
        self.val_loader = build_detection_test_loader(cfg, "lary_val", mapper=mapper)

        super().__init__(cfg)  # call DefaultTrainer constructor

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_hooks(self):
        hooks_list = super().build_hooks()

        # Insert validation loss hook 
        if self.val_loader is not None:
            hooks_list.insert(-1, ValidationLossHook(
                model=self.model,
                val_loader=self.val_loader,
                patience=self.patience,
                checkpointer=self.checkpointer,
                results_folder=self.results_folder,
                print_freq=100
            ))
        return hooks_list

    def run_step(self):
        """
        Standard training step
        """
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)

        data = next(self._data_loader_iter)

        # --- Forward + backward ---
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        total_loss = losses.item()
        storage = get_event_storage()
        storage.put_scalar("total_loss", total_loss)
        if self.iter % 20 == 0:
            print(f"[Iter {self.iter}] Training loss: {total_loss:.4f}")


# ---  Metrics ---#
### only dice nad ap is reported in the paper###
def calculate_advanced_metrics(model, data_loader, coco_results, num_classes=6):
    model.eval()
    stats = {i: {"dice": [], "iou": [], "hd95": [], "rel_hd95": [], "detected": 0} for i in range(num_classes)}
    total_images = 0
    all6_detected_images = 0
    target_size = (1024, 1024) 
    
    print(f"Calculating DICE, IoU, HD95, Relative HD95, and detection rates at {target_size} resolution...")
    
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            
            for i, input_data in enumerate(inputs):
                total_images += 1
                pred_instances = outputs[i]["instances"].to("cpu")
                pred_instances = pred_instances[pred_instances.scores > 0.5]
                pred_classes = pred_instances.pred_classes.numpy().tolist() if len(pred_instances) > 0 else []
                
                # Track per-class detection
                detected_classes = set(pred_classes)
                for class_id in range(num_classes):
                    if class_id in detected_classes:
                        stats[class_id]["detected"] += 1
                
                # Track all-6 detection
                if len(detected_classes) == num_classes:
                    all6_detected_images += 1

                gt_instances = input_data["instances"].to("cpu")
                for class_id in range(num_classes):
                    # --- Ground Truth Mask ---
                    gt_mask_np = np.zeros(target_size, dtype=bool)
                    idx_gt = gt_instances.gt_classes == class_id
                    diag = 0
                    
                    if idx_gt.any():
                        m = gt_instances.gt_masks
                        m_tensor = m.tensor[idx_gt] if hasattr(m, "tensor") else m[idx_gt]
                        m_resized = F.interpolate(m_tensor.float().unsqueeze(0), size=target_size, mode="nearest")[0]
                        gt_mask_np = m_resized.sum(dim=0).numpy() > 0
                        
                        coords = np.argwhere(gt_mask_np)
                        if coords.size > 0:
                            y_min, x_min = coords.min(axis=0)
                            y_max, x_max = coords.max(axis=0)
                            diag = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)

                    # --- Prediction Mask ---
                    pred_mask_np = np.zeros(target_size, dtype=bool)
                    idx_pred = pred_instances.pred_classes == class_id
                    if idx_pred.any():
                        p_m = pred_instances.pred_masks[idx_pred]
                        if p_m.shape[1:] != target_size:
                            p_m = F.interpolate(p_m.float().unsqueeze(0), size=target_size, mode="nearest")[0]
                        pred_mask_np = p_m.sum(dim=0).numpy() > 0

                    # --- Compute Metrics ---
                    if np.any(gt_mask_np) and np.any(pred_mask_np):
                        intersection = np.logical_and(gt_mask_np, pred_mask_np).sum()
                        union = np.logical_or(gt_mask_np, pred_mask_np).sum()
                        dice = (2.0 * intersection) / (gt_mask_np.sum() + pred_mask_np.sum())
                        iou = intersection / union if union > 0 else 0
                        
                        stats[class_id]["dice"].append(dice)
                        stats[class_id]["iou"].append(iou)
                        
                        try:
                            raw_hd95 = metric.binary.hd95(pred_mask_np, gt_mask_np)
                            stats[class_id]["hd95"].append(raw_hd95)
                            if diag > 0:
                                rel_hd95 = (raw_hd95 / diag) * 100
                                stats[class_id]["rel_hd95"].append(rel_hd95)
                        except:
                            pass

    # --- Get Class Names ---
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    class_names = metadata.get("thing_classes", [f"Class {i}" for i in range(num_classes)])

    # --- Final Summary ---
    summary = []
    for class_id in range(num_classes):
        c_name = class_names[class_id]
        ap_key = f"AP-{c_name}"
        class_ap50 = coco_results.get('segm', {}).get(ap_key, 0.0)
        
        detect_recall = stats[class_id]["detected"] / total_images if total_images > 0 else 0
        
        summary.append([
            c_name,
            class_ap50,
            np.mean(stats[class_id]["dice"]) if stats[class_id]["dice"] else 0,
            np.mean(stats[class_id]["iou"]) if stats[class_id]["iou"] else 0,
            np.mean(stats[class_id]["rel_hd95"]) if stats[class_id]["rel_hd95"] else np.nan,
            detect_recall
        ])
    
    # --- Overall all-6-class detection rate ---
    all6_rate = all6_detected_images / total_images if total_images > 0 else 0
    print(f"All-6-class detection rate: {all6_rate:.4f}")

    return summary, all6_rate

# -----------------------------
# Load weights
# -----------------------------

def load_trained_model(checkpoint_path):
    # Recreate config 
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(
        "./Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
    )

    thing_classes = list(Mask2FormerDataset.LABEL_MAP.keys())
    cfg.MODEL.MASK_FORMER.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(thing_classes)

    # Build model
    model = build_model(cfg)
    
    # Load checkpoint weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

# -----------------------------
# Train
# -----------------------------
val_json   = "./lary_seg/data/eval"
test_json  = "./lary_seg/data/test"

DatasetCatalog.register("lary_val", lambda: get_lary_dicts(val_json))
DatasetCatalog.register("lary_test", lambda: get_lary_dicts(test_json))

thing_classes = list(Mask2FormerDataset.LABEL_MAP.keys())
MetadataCatalog.get("lary_val").set(thing_classes=thing_classes)
MetadataCatalog.get("lary_test").set(thing_classes=thing_classes)


# -----------------------------
# Detectron2 Config
# -----------------------------
cfg = get_cfg()

# add Mask2Former & DeepLab config schemas
add_deeplab_config(cfg)
add_maskformer2_config(cfg)

# merge the base YAML
cfg.merge_from_file(
    "./Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
)

# dataset and number of classes
cfg.DATASETS.TEST = ("lary_val",)
cfg.MODEL.MASK_FORMER.NUM_CLASSES = len(thing_classes)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(thing_classes)


# pretrained weights (official Mask2Former URL)
cfg.MODEL.WEIGHTS = "./Mask2Former/model_final_3c8ec9.pkl"

#  dataloader & solver
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.TEST.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.MAX_ITER = 8000
cfg.SOLVER.WEIGHT_DECAY = 0.001
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

#  output folder
cfg.MODEL.MASK_FORMER.INSTANCES_ON = True
cfg.MODEL.MASK_FORMER.SEM_SEG_POSTPROCESSING = True

# --- Define conditions ---
conditions = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
# --- Initialize metric lists ---
dice_means, iou_means, rhd95_means, ap_means, det_rate = [], [], [], [], []

for cond in conditions:
    # Set folder paths based on condition
    percentage = int(cond * 100)
    train_folder_r = f"./lary_seg/data/train_0614/train_{percentage}"
    dataset_name = f"lary_train_{percentage}"
    output_dir = f"./mask2former_output_0614/train_{percentage}"

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "reduced_dataset.json")
    create_reduced_dataset(
        original_folder="./lary_seg/data/train",
        target_folder=train_folder_r,
        reduction_ratio=cond,
        seed=100,
        record_json=json_path
    )

    os.makedirs(output_dir, exist_ok=True)

    # --- Register dataset ---
    DatasetCatalog.register(dataset_name, lambda folder=train_folder_r: get_lary_dicts(folder))
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
    cfg.DATASETS.TRAIN = [dataset_name]
    cfg.OUTPUT_DIR = output_dir

    # --- Train ---
    trainer = TrainerWithEarlyStopping(cfg, patience=10, results_folder=output_dir)
    try:
        trainer.resume_or_load(resume=False)
        trainer.train()
    except StopTrainingException:
        print("Early stopping: training exited cleanly.")

    # ---------- Load Best Weights ----------
    print("--- Loading Supervised Model Weights for Evaluation ---")
    ### final_model is used here as it has much better performance, especially for low-data regimes ###
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "final_model.pth") 
    
    if os.path.exists(best_model_path):
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(trainer.model).load(best_model_path)
    else:
        print(f"⚠️ Warning: {best_model_path} not found. Evaluating with final weights.")

    # --- Evaluate ---
    evaluator = COCOEvaluator("lary_test", cfg, False, output_dir=output_dir)
    test_loader = build_detection_test_loader(cfg, "lary_test", mapper=mapper)
    coco_results = inference_on_dataset(trainer.model, test_loader, evaluator)

    # --- Calculate metrics ---
    final_stats, all6_rate = calculate_advanced_metrics(trainer.model, test_loader, coco_results)

    # --- Aggregate metrics ---
    mean_dice = np.nanmean([row[2] for row in final_stats])
    mean_iou = np.nanmean([row[3] for row in final_stats])
    mean_rhd95 = np.nanmean([row[4] for row in final_stats])
    mean_ap = coco_results["segm"]["AP"] / 100.0

    dice_means.append(mean_dice)
    iou_means.append(mean_iou)
    rhd95_means.append(mean_rhd95)
    ap_means.append(mean_ap)
    det_rate.append(all6_rate)

    # --- Save detailed table ---
    headers = ["Structure", "AP (Det)", "Dice (DSC)", "IoU (Jaccard)", "R-HD95 (px)", "Detect Recall"]
    table_data = [
        [row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
        for row in final_stats
    ]
    # Append a row for all-6 detection rate
    table_data.append(["All6_detected_rate", "-", "-", "-", "-", all6_rate])

    table_str = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")
    print(f"\n=== Metrics for {dataset_name} ===")
    print(table_str)
    print(f"Mean Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, R-HD95: {mean_rhd95:.4f}, AP: {mean_ap:.2f}, All6 Rate: {all6_rate:.4f}")

    report_file = os.path.join(output_dir, f"detailed_metrics.txt")
    with open(report_file, "w") as f:
        f.write(f"=== Metrics for {dataset_name} ===\n")
        f.write(table_str + "\n")
        f.write(f"Mean Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, R-HD95: {mean_rhd95:.4f}, AP: {mean_ap:.2f}, All6 Rate: {all6_rate:.4f}\n")

# --- Save all metrics to JSON ---
metrics_dict = {
    "conditions": conditions,
    "dice": dice_means,
    "iou": iou_means,
    "rhd95": rhd95_means,
    "ap": ap_means,
    "all6_rate":  det_rate  # store the per-condition All6 rate
}
json_file = os.path.join(cfg.OUTPUT_DIR, "all_metrics.json")
with open(json_file, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print(f"Saved all metrics to {json_file}")

