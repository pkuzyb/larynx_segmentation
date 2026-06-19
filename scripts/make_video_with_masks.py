import cv2
import numpy as np
from tqdm import tqdm 

# Color Mapping Configuration (0-Indexed Channels matching standard layout)
COLOR_MAPPING = {
    0: (128, 128, 128),  # c0: Thyroid Cartilage -> Gray
    1: (255, 0, 255),    # c1: Laryngeal Ventricle -> Purple
    2: (0, 255, 0),      # c2: True Vocal Fold -> Green
    3: (0, 165, 255),    # c3: Arytenoid Cartilage -> Orange
    4: (220, 20, 60),    # c4: Epiglottis -> Steel Blue / Crimson Variant
    5: (71, 99, 255)     # c5: Ventricular Folds (False Folds) -> Coral Red
}

def stream_and_process_video(video_path, npz_path, output_path, mask_threshold=0.3):
    """
    Streams and processes video frames by unpacking bit-packed channel masks 
    on-the-fly to reconstruct native 1024x1024 pixel spaces cleanly.
    """
    video = cv2.VideoCapture(video_path)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    if num_frames == 0 or fps == 0:
        raise ValueError(f"Error: Unable to open video or video file is empty: {video_path}")

    print("Opening compressed mask archive...")
    npz_data = np.load(npz_path)
        
    # Setup VideoWriter for a standalone 1024x1024 frame canvas layout
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_path, fourcc, fps, (1024, 1024))
    
    try:
        for i in tqdm(range(num_frames), desc="Processing pipeline"):
            ret, frame = video.read()
            if not ret:
                print(f"\nWarning: Video ended prematurely or frame is corrupt at index {i}")
                break  
            
            # Upscale single raw frame to uniform 1024x1024 dimensions
            frame_resized = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            
            # Create a clean canvas to hold mask colors for alpha blending
            mask_overlay = np.zeros_like(frame_resized)
            
            # Extract and apply channels sequentially
            for channel_idx, color_rgb in COLOR_MAPPING.items():
                key_name = f"f{i}_c{channel_idx}"
                
                if key_name in npz_data:
                    packed_layer = npz_data[key_name]
                    
                    # --- DECODE BIT-PACKED MASK ---
                    # Unpack 131,072 bytes back into 1,048,576 binary pixels and reshape to square grid
                    mask_layer = np.unpackbits(packed_layer).reshape(1024, 1024)
                    # ------------------------------

                    # Convert color tuple layout from standard RGB to OpenCV BGR
                    color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.uint8)
                    
                    # Paint activations exceeding our confidence floor threshold onto the overlay canvas
                    activation_indices = mask_layer > mask_threshold
                    mask_overlay[activation_indices] = color_bgr
                
            # Blend raw frame (70% weight) and mask canvas (30% weight) for semi-transparency
            blended_frame = cv2.addWeighted(frame_resized, 0.7, mask_overlay, 0.3, 0)
            
            # Flush directly to disk (this preserves server RAM)
            out.write(blended_frame)
            
    finally:
        video.release() 
        out.release()
        npz_data.close()
        
    print(f"\nSuccess: Overlaid visualization sequence exported to -> {output_path}")

if __name__ == "__main__":
    input_video = "./videos/mando_01/mando_01_rtmri_07_Tone_r1.avi"
    input_npz_masks = "./masks/train_100/mando_01_rtmri_07_Tone_r1_masks.npz"
    output_composite_video = "./mando_01_rtmri_07_Tone_r1_masks_overlaid.mp4"
    
    try:
        stream_and_process_video(
            video_path=input_video,
            npz_path=input_npz_masks,
            output_path=output_composite_video,
            mask_threshold=0.3
        )
        
    except FileNotFoundError as error:
        print(f"Pipeline Execution Aborted: Missing required files -> {error}")
    except Exception as general_error:
        print(f"Encountered unexpected execution error: {general_error}")