# Larynx segmentation for mid-sagittal speech production real-time MRI
<video src="https://raw.githubusercontent.com/pkuzyb/larynx_segmentation/main/inference_examples/videos_with_masks/lary_seg_demo.mp4" width="512" controls muted loop autoplay></video>

This repository contains data, checkpoints, and code for larynx segmentation in mid-sagittal real-time speech production MRI.

---

## Dataset
The dataset consists of JPG-format MRI frames and corresponding JSON-format segmentation annotations:

```text
./data
```

---

## Setup
The setup script installs required dependencies, including Detectron2.

```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## Training

Train Mask2Former models:

```bash
python scripts/train_maskformer.py
```

Train teacher-student semi-supervised learning models:

```bash
python scripts/train_ssl.py
```

---
## Inference
Pretrained checkpoints for both supervised and semi-supervised models trained under 25% and 100% labeled-data conditions:
```text
./models/
```
To make inferences on new MRI frames:

```bash
python inference.py
```
Sample videos and predicted segmentation masks:
```text
./inference_examples/videos
./inference_examples/masks
```

---
## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{zhang2026larynx,
  author={Zhang, Yubin and Shi, Xuan and Huang, Kevin and Kumar, Prakash and Lee, Kevin and Goldstein, Louis and Krsina, Nayak and Narayanan, Shrikanth},
  title     = {Larynx segmentation in mid-sagittal speech production real-time MRI},
  booktitle = {Proceedings of Interspeech},
  year      = {2026}
}
```

---
