import os
import json
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import sys
from pycocotools import mask as mask_util
from tabulate import tabulate 
import torch.nn.functional as F
from medpy import metric

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import Instances, Boxes, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from mask2former import add_maskformer2_config

from dataset import Mask2FormerDataset, UnlabeledMask2FormerDataset 


sys.path.append("./Mask2Former")

# -----------------------------
# Dataset Registration
# -----------------------------
def get_lary_dicts(json_folder):
    dataset = Mask2FormerDataset(json_folder,resize_to=1024)
    return [record for record in dataset]


def get_unlabeled_dicts(img_folder):
    dataset = UnlabeledMask2FormerDataset(img_folder, resize_to=1024)
    records = []
    
    for i in range(len(dataset)):
        record = dataset[i]
        if record is not None:
            records.append(record)
            
    print(f"DEBUG: Returning {len(records)} pre-resized unlabeled records from Class.")
    return records

# -----------------------------
# Data Augmentation Mapper
# -----------------------------
def mapper(dataset_dict, is_strong=False):
    """
    Loads and resizes images to 1024x1024 (done here not in dataset.py).
    Utilizes pre-rasterized smooth 1024 masks from dataset.py.
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    
    # 1. Read Raw Image 
    image_raw = utils.read_image(dataset_dict["file_name"], format="BGR")
    orig_h, orig_w = image_raw.shape[:2]

    # 2. Rescale to 1024
    target_size = 1024
    scale = target_size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    
    # Create the 'Weak' version - Linear interpolation for smooth gray levels
    image_weak = cv2.resize(image_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype("float32") 

    # 3. Create Strong Version for Student
    if is_strong:
        img = image_weak.copy()
        
        # --- Rician Noise (simulating MRI acquisition grain) ---
        if np.random.rand() > 0.5:
            sigma = np.random.uniform(25, 35) 
            noise_real = np.random.normal(0, sigma, img.shape)
            noise_imag = np.random.normal(0, sigma, img.shape)
            img = np.sqrt((img + noise_real)**2 + noise_imag**2).astype(np.float32)

        # --- Gaussian Blur (softening the edges) ---
        if np.random.rand() > 0.5:
            img = cv2.GaussianBlur(img, (7, 7), 0)


        dataset_dict["image"] = torch.from_numpy(img.transpose(2, 0, 1))
    else:
        dataset_dict["image"] = torch.from_numpy(image_weak.transpose(2, 0, 1))

    # Store image_weak for the Teacher (Pseudo-label generation)
    dataset_dict["image_weak"] = torch.from_numpy(image_weak.transpose(2, 0, 1))

    # Process ground truth (Already 1024-scale from dataset.py)
    annos = [obj for obj in dataset_dict.pop("annotations", []) if obj.get("iscrowd", 0) == 0]
    masks, boxes, classes = [], [], []

    for ann in annos:
        #  The masks are already 1024. Do not cv2.resize it.
        mask = mask_util.decode(ann["segmentation"]).astype(np.uint8)
        
        # Also, Bbox is already scaled in dataset.py
        x0, y0, w_b, h_b = ann["bbox"]
        bbox = [x0, y0, x0 + w_b, y0 + h_b]
        
        masks.append(torch.from_numpy(mask))
        boxes.append(torch.tensor(bbox, dtype=torch.float32))
        classes.append(ann["category_id"])

    # 5. Create Instances
    instances = Instances((new_h, new_w))
    if len(masks) > 0:
        instances.gt_boxes = Boxes(torch.stack(boxes))
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_masks = torch.stack(masks).to(torch.uint8) 
    else:
        instances.gt_boxes = Boxes(torch.empty((0, 4)))
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        instances.gt_masks = torch.empty((0, new_h, new_w), dtype=torch.uint8)

    dataset_dict["instances"] = instances
    return dataset_dict

# -----------------------------
# EMA Hook
# -----------------------------
class EMAWeightUpdateHook(hooks.HookBase):
    def __init__(self, student_model, teacher_model, alpha=0.999):
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha

    def after_step(self):
        with torch.no_grad():
            s_state = self.student.state_dict()
            t_state = self.teacher.state_dict()
            for k in s_state.keys():
                t_state[k] = self.alpha * t_state[k] + (1.0 - self.alpha) * s_state[k]
            self.teacher.load_state_dict(t_state)

# --- Custom Exception for Early Stopping ---
class StopTrainingException(Exception):
    """Custom exception to stop training cleanly"""
    pass

# --- Validation Loss & Early Stopping Hook ---
class SSLValidationLossHook(hooks.HookBase):
    def __init__(self, val_loader, patience= 5, eval_period=10):
        self.val_loader = val_loader
        self.patience = patience
        self.eval_period = eval_period
        self.best_val_loss = float("inf")
        self.no_improve_count = 0
        self.zero_pl_streak = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.eval_period != 0:
            return


        val_losses = []
        self.trainer.model.train() # Switch to train mode to get loss dict
 
        
        print(f"\n--- [Iter {self.trainer.iter}] Running Validation Evaluation ---")
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss_dict = self.trainer.model(batch)
                total_loss = sum(loss_dict.values()).item()
                val_losses.append(total_loss)

        avg_val_loss = np.mean(val_losses)
        storage = get_event_storage()
        storage.put_scalar("val_loss", avg_val_loss)

        # Get the latest pseudo-label count from storage
        pl_count = storage.history("num_pseudo_labels").latest() if "num_pseudo_labels" in storage.histories() else 0

        # Retrieve Training Losses and PL Count from storage
        train_labeled = storage.history("loss_labeled").latest() if "loss_labeled" in storage.histories() else 0.0
        train_unlabeled = storage.history("loss_unlabeled").latest() if "loss_unlabeled" in storage.histories() else 0.0
        pl_count = storage.history("num_pseudo_labels").latest() if "num_pseudo_labels" in storage.histories() else 0

        # Print everything together
        print("="*80)
        print(f"📊 EVALUATION SUMMARY AT ITER {self.trainer.iter}")
        print(f"TRAIN: Labeled Loss: {train_labeled:.4f} | Unlabeled Loss: {train_unlabeled:.4f} | PL Count: {int(pl_count)}")
        print(f"EVAL:  Total Val Loss: {avg_val_loss:.4f} (Best: {self.best_val_loss:.4f})")
        print("="*80)

        # Teacher Starvation Check
        # If the Teacher found 0 labels in this batch
        if pl_count == 0:
            self.zero_pl_streak += 1
        else:
            self.zero_pl_streak = 0 # Reset if Teacher finally teaches something

        # 500 iterations / eval_period(50) = 10 checks
        max_starvation_checks = 500 // self.eval_period 
        
        if self.zero_pl_streak >= max_starvation_checks:
            print(f"SSL STARVATION: Teacher has taught nothing for 500 iterations. Stopping early.")
            raise StopTrainingException()
        
        # Early Stopping Logic
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.no_improve_count = 0
            self.trainer.checkpointer.save("best_ssl_model")
        else:
            self.no_improve_count += 1
            print(f"No improvement for {self.no_improve_count}/{self.patience} checks.")

        if self.no_improve_count >= self.patience:
            print(f"Early stopping triggered.")
            raise StopTrainingException()
        
class SSLTrainer(DefaultTrainer):
    def __init__(self, cfg):
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Setup Teacher Model
        self.teacher_model = build_model(cfg)
        self.teacher_model.to(self.device)
        
        # Setup Unlabeled Data Loader
        unlabeled_dicts = get_unlabeled_dicts(unlabeled_dir)
        print(f"SSLTrainer: Initialized with {len(unlabeled_dicts)} unlabeled records.")
        
        self.unlabeled_loader = iter(build_detection_train_loader(
            cfg, 
            dataset=unlabeled_dicts, 
            mapper=lambda x: mapper(x, is_strong=True)
        ))
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=lambda x: mapper(x, is_strong=False))

    def build_hooks(self):
        hooks_list = super().build_hooks()
        hooks_list.append(EMAWeightUpdateHook(self.model, self.teacher_model))
        val_loader = build_detection_test_loader(self.cfg, "lary_val", mapper=mapper)
        hooks_list.append(SSLValidationLossHook(
            val_loader=val_loader, 
            patience=5, 
            eval_period=50
        ))
        return hooks_list
            
    def run_step(self):
        self.model.train()
        storage = get_event_storage()
        
        # -----------------------------------------------------------
        # LABELED STEP (STUDENT)
        # -----------------------------------------------------------
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self.data_loader)
        try:
            data_labeled = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data_labeled = next(self._data_loader_iter)
            
        loss_dict_labeled = self.model(data_labeled)
        labeled_loss_val = sum(loss_dict_labeled.values())
        
        # -----------------------------------------------------------
        # TEACHER INFERENCE (UNLABELED - WEAK IMAGE)
        # -----------------------------------------------------------
        try:
            data_unlabeled = next(self.unlabeled_loader)
        except StopIteration:
            unlabeled_dicts = get_unlabeled_dicts(unlabeled_dir) 
            self.unlabeled_loader = iter(build_detection_train_loader(
                self.cfg, dataset=unlabeled_dicts, mapper=lambda x: mapper(x, is_strong=True)
            ))
            data_unlabeled = next(self.unlabeled_loader)

        with torch.no_grad():
            self.teacher_model.eval()
            teacher_inputs = []
            for d in data_unlabeled:
                h, w = d["image_weak"].shape[1:] 
                teacher_inputs.append({
                    "image": d["image_weak"].to(self.device), 
                    "height": h, 
                    "width": w   
                })
            teacher_preds = self.teacher_model(teacher_inputs)


        # -----------------------------------------------------------
        # PROCESS PSEUDO-LABELS 
        # -----------------------------------------------------------
        valid_unlabeled_batch = []
        
        # Number of pseudo labels
        num_pl = 0
        max_batch_score = 0.0
        pred_class_names_batch = [] 
        
        for i, item in enumerate(data_unlabeled):
            inst = teacher_preds[i]["instances"].to("cpu")

            if len(inst) > 0:
                max_batch_score = max(max_batch_score, inst.scores.max().item())

            # Threshold check 
            keep = inst.scores > 0.85
            filtered_inst = inst[keep].to(self.device)

            # Map filtered classes to names 
            if len(filtered_inst) > 0:
                pred_class_ids = filtered_inst.pred_classes.tolist()
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
                class_names = metadata.get("thing_classes", [f"Class {i}" for i in range(len(pred_class_ids))])
                pred_class_names = [class_names[c] for c in pred_class_ids]
                pred_class_names_batch.extend(pred_class_names)

                filtered_inst.gt_classes = filtered_inst.pred_classes
                filtered_inst.gt_masks = filtered_inst.pred_masks.detach().to(torch.uint8)
                
                # gt_boxes 
                if filtered_inst.has("pred_boxes"):
                    filtered_inst.gt_boxes = filtered_inst.pred_boxes
                else:
                    from detectron2.structures import BitMasks
                    filtered_inst.gt_boxes = BitMasks(filtered_inst.gt_masks).get_bounding_boxes()

                # Remove inference fields so Student treats this as GT
                fields_to_remove = ["pred_masks", "pred_boxes", "pred_classes", "scores"]
                for field in fields_to_remove:
                    if filtered_inst.has(field):
                        filtered_inst.remove(field)
                
                item["instances"] = filtered_inst
                img_h, img_w = item["image"].shape[1:] 
                inst_h, inst_w = filtered_inst.image_size

                if img_h != inst_h or img_w != inst_w:
                    print(f"SCALE MISMATCH: Image is {img_h}x{img_w}, but Instances are {inst_h}x{inst_w}")
                
                valid_unlabeled_batch.append(item)
                num_pl += len(filtered_inst)
                
        # -----------------------------------------------------------
        # UNLABELED STEP (STUDENT - STRONG IMAGE)
        # -----------------------------------------------------------
        unlabeled_loss_val = torch.tensor(0.0).to(self.device)
        
        # THE EMPTY FIX: Skip student forward if teacher found nothing
        if len(valid_unlabeled_batch) > 0:
            loss_dict_unlabeled = self.model(valid_unlabeled_batch)
            unlabeled_loss_val = sum(loss_dict_unlabeled.values())

        # Total combined training loss
        total_loss = labeled_loss_val + 0.5 * unlabeled_loss_val

        # -----------------------------------------------------------
        # OPTIMIZATION & LOGGING
        # -----------------------------------------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        storage.put_scalar("loss_labeled", labeled_loss_val.item())
        storage.put_scalar("loss_unlabeled", unlabeled_loss_val.item())
        storage.put_scalar("total_train_loss", total_loss.item())
        storage.put_scalar("num_pseudo_labels", num_pl)

        if self.iter % 20 == 0:
            print(
                f"[Iter {self.iter}] Labeled: {labeled_loss_val.item():.3f} | "
                f"Unlabeled: {unlabeled_loss_val.item():.3f} | num_pseudo_labels: {num_pl} | "
                f"Scores: [{max_batch_score:.2f}] | Teacher Pred (score>0.85): {pred_class_names_batch}")

                    
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
# Detectron2 Config
# -----------------------------
val_json   = "./lary_seg/data/eval"
test_json  = "./lary_seg/data/test"
unlabeled_dir = "./lary_seg/data/unlabelled"

DatasetCatalog.register("lary_val", lambda: get_lary_dicts(val_json))
DatasetCatalog.register("lary_test", lambda: get_lary_dicts(test_json))
DatasetCatalog.register("lary_unlabeled", lambda: get_unlabeled_dicts(unlabeled_dir))

thing_classes = list(Mask2FormerDataset.LABEL_MAP.keys())
for d in ["lary_train", "lary_val", "lary_test", "lary_unlabeled"]:
    MetadataCatalog.get(d).set(thing_classes=thing_classes)

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("./Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")

cfg.DATASETS.TEST = ("lary_val",)
cfg.MODEL.MASK_FORMER.NUM_CLASSES = len(thing_classes)
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(thing_classes)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.TEST.IMS_PER_BATCH = 4

cfg.SOLVER.BASE_LR = 1e-4
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
cfg.SOLVER.MAX_ITER = 10000
cfg.OUTPUT_DIR = "./mask2former_ssl_output_0614"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# -----------------------------
# CONDITIONS
# -----------------------------
conditions = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]

# -----------------------------
# Initialize metric lists
# -----------------------------
dice_means, iou_means, rhd95_means, ap_means, det_rate = [], [], [], [], []

# -----------------------------
# SSL LOOP
# -----------------------------
for cond in conditions:
    pct = int(cond * 100)
    print(f"=== SSL Training: {pct}% labeled ===")
    train_folder = f"./lary_seg/data/train_0614/train_{pct}"
    dataset_name = f"lary_train_{pct}"
    ssl_output = f"./mask2former_ssl_output_0614/train_{pct}"
    nonssl_weights = f"./mask2former_output_0614/train_{pct}/final_model.pth"

    os.makedirs(ssl_output, exist_ok=True)

    # -----------------------------
    # Dataset registration
    # -----------------------------
    if dataset_name not in DatasetCatalog.list():
        DatasetCatalog.register(
            dataset_name,
            lambda folder=train_folder: get_lary_dicts(folder)
        )
        MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)

    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.OUTPUT_DIR = ssl_output

    # -----------------------------
    # Load teacher (non-SSL) weights
    # -----------------------------
    if not os.path.exists(nonssl_weights):
        raise FileNotFoundError(f"Missing non-SSL weights: {nonssl_weights}")
    cfg.MODEL.WEIGHTS = nonssl_weights
    print("Teacher initialized from:", nonssl_weights)

    # -----------------------------
    # SSL Trainer
    # -----------------------------
    trainer = SSLTrainer(cfg)
    try:
        trainer.resume_or_load(resume=False)
        # copy student → teacher at start
        trainer.teacher_model.load_state_dict(trainer.model.state_dict())
        trainer.train()
    except StopTrainingException:
        print("Early stop triggered (SSL).")


    print("--- Loading Best SSL Model Weights for Evaluation ---")
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_ssl_model.pth")
    if os.path.exists(best_model_path):
        from detectron2.checkpoint import DetectionCheckpointer
        DetectionCheckpointer(trainer.model).load(best_model_path)
    else:
        print("Warning: best_ssl_model.pth not found, using final weights instead.")
        
    # -----------------------------
    # Evaluation
    # -----------------------------
    evaluator = COCOEvaluator("lary_test", cfg, False, output_dir=ssl_output)
    test_loader = build_detection_test_loader(cfg, "lary_test", mapper=mapper)
    coco_results = inference_on_dataset(trainer.model, test_loader, evaluator)

    # -----------------------------
    # Calculate metrics (including All-6-class detection rate)
    # -----------------------------
    final_stats, all6_rate = calculate_advanced_metrics(trainer.model, test_loader, coco_results)

    # -----------------------------
    # Aggregate metrics
    # -----------------------------
    mean_dice = np.nanmean([row[2] for row in final_stats])
    mean_iou = np.nanmean([row[3] for row in final_stats])
    mean_rhd95 = np.nanmean([row[4] for row in final_stats])
    mean_ap = coco_results["segm"]["AP"] / 100.0

    dice_means.append(mean_dice)
    iou_means.append(mean_iou)
    rhd95_means.append(mean_rhd95)
    ap_means.append(mean_ap)
    det_rate.append(all6_rate)

    # -----------------------------
    # Save detailed metrics table
    # -----------------------------
    headers = ["Structure", "AP", "Dice", "IoU", "R-HD95", "Detect Recall"]
    table_data = [[row[0], row[1], row[2], row[3], row[4], row[5]] for row in final_stats]
    table_data.append(["All6_detected_rate", "-", "-", "-", "-", all6_rate])

    table_str = tabulate(
        table_data,
        headers=headers,
        tablefmt="grid",
        floatfmt=".4f"
    )

    print(f"\n=== Metrics for {dataset_name} ===")
    print(table_str)
    print(f"Mean Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, R-HD95: {mean_rhd95:.4f}, AP: {mean_ap:.2f}, All6 Rate: {all6_rate:.4f}")


    report_file = os.path.join(ssl_output, "detailed_metrics.txt")
    with open(report_file, "w") as f:
        f.write(table_str + "\n")
        f.write(f"Mean Dice: {mean_dice:.4f}, IoU: {mean_iou:.4f}, R-HD95: {mean_rhd95:.4f}, "
                f"AP: {mean_ap:.2f}, All6 Rate: {all6_rate:.4f}\n")

# -----------------------------
# SAVE ALL SSL METRICS
# -----------------------------
metrics_dict = {
    "conditions": conditions,
    "dice": dice_means,
    "iou": iou_means,
    "rhd95": rhd95_means,
    "ap": ap_means,
    "all6_rate": det_rate
}

metrics_file = os.path.join(ssl_output,"all_metrics_ssl.json")
with open(metrics_file, "w") as f:
    json.dump(metrics_dict, f, indent=4)
print("Saved SSL metrics →", metrics_file)
