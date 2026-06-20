import os
import sys
import cv2
import torch
import numpy as np
import gc
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# Append the Mask2Former folder
sys.path.append("./Mask2Former")
from mask2former import add_maskformer2_config

# -----------------------------
# Configuration
# -----------------------------
LABEL_MAP = {
    'thyroid_cartilage': 0, 'ventricle': 1, 'vocal_folds': 2,
    'arytenoid_cartilage': 3, 'epiglottis': 4, 'ventricular_folds': 5,
}
NUM_CLASSES = len(LABEL_MAP)

def prepare_model(weights_path, device='cuda'):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("./Mask2Former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    cfg.MODEL.MASK_FORMER.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE = str(device) 

    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_path)
    model.eval()
    return model

def predict_frames_generator(video_path, model, device, batch_size=8):
    video = cv2.VideoCapture(video_path)
    nframe = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with torch.no_grad():
        for i in range(0, nframe, batch_size):
            batch_frames = []
            for _ in range(batch_size):
                ret, frame = video.read()
                if not ret: break
                batch_frames.append(cv2.resize(frame, (1024, 1024)))
            
            if not batch_frames: break

            inputs = []
            for frame in batch_frames:
                image_tensor = torch.as_tensor(frame.transpose(2, 0, 1), dtype=torch.float32).to(device)
                inputs.append({"image": image_tensor, "height": 1024, "width": 1024})

            outputs = model(inputs)
            yield [out["instances"].to("cpu") for out in outputs]
            
            del inputs, outputs
    video.release()

# -----------------------------
# Mask Extraction 
# -----------------------------
def extract_masks(output_base, video_path, model, device_obj, condition_name):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get total frames to calculate total batches for the progress bar
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    batch_size = 8
    total_batches = (total_frames + batch_size - 1) // batch_size

    print(f"  🎬 [{condition_name}] Processing: {video_name}")

    packed_masks = {}
    batch_gen = predict_frames_generator(video_path, model, device_obj, batch_size=batch_size)
    
    frame_idx = 0
    
    # Wrap the generator with tqdm for a live progress bar
    for batch_instances in tqdm(batch_gen, total=total_batches, desc="    Frames", unit="batch"):
        for inst in batch_instances:
            for c in range(NUM_CLASSES):
                class_indices = (inst.pred_classes == c)
                
                if torch.any(class_indices):
                    # Pick the instance with the highest confidence score
                    best_idx = torch.argmax(inst.scores[class_indices]).item()
                    mask_np = inst.pred_masks[class_indices][best_idx].numpy().astype(bool)
                    
                    # Bit-pack (saves ~8x storage space)
                    packed_masks[f"f{frame_idx}_c{c}"] = np.packbits(mask_np)
            
            frame_idx += 1
        torch.cuda.empty_cache()

    # Save 
    save_dir = os.path.join(output_base, "masks", condition_name)
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{video_name}_masks.npz")
    
    np.savez_compressed(npz_path, **packed_masks)
    print(f"  Saved: {os.path.basename(npz_path)} ({os.path.getsize(npz_path)/1024/1024:.2f} MB)")
    
    del packed_masks
    gc.collect()

if __name__ == "__main__":
    ROOT_WEIGHTS = './XXX'
    CONDITIONS = ['train_25', 'train_100']
    BASE_OUTPUT = './lary_seg'
    VIDEO_ROOT = os.path.join(BASE_OUTPUT, 'videos')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for cond in CONDITIONS:
        weights_file = os.path.join(ROOT_WEIGHTS, cond, 'best_ssl_model.pth')
        
        if not os.path.exists(weights_file):
            print(f"Weights not found for {cond}, skipping...")
            continue

        print(f"\n Loading Model for Condition: {cond}")
        model = prepare_model(weights_file, device=DEVICE)

        # Process all videos for this specific model condition
        for root, _, files in os.walk(VIDEO_ROOT):
            for video_file in [f for f in files if f.endswith(('.avi', '.mp4'))]:
                v_path = os.path.join(root, video_file)
                try:
                    extract_masks(BASE_OUTPUT, v_path, model, DEVICE, cond)
                except Exception as e:
                    print(f"Error in {video_file} with {cond}: {e}")
                    continue
        
        # Free model from VRAM before switching conditions
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\n✨ All conditions and videos processed.")