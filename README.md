# Larynx segmentation in mid-sagittal speech production real-time MRI

<p align="center">
  <img src="inference/videos_with_masks/lary_seg_demo.gif" alt="Larynx Segmentation Demo" style="width: 70%;">
</p>

This repository contains data, checkpoints, and code for larynx segmentation in mid-sagittal real-time speech production MRI.

---

## Dataset
The dataset consists of JPG-format rt-MRI frames and corresponding JSON-format segmentation annotations organized as follows:

```text
data/
├── train/              # Labeled training images and annotations
├── eval/               # Validation set used for model selection
├── test/               # Held-out test set for final evaluation
├── train_data_split/   # Labeled-data subsets generated from one random sampling seed (1%, 2.5%, 5%, 7.5%, 10%, 25%, 50%, 75%, and 100% labeled data)
└── unlabelled/         # Unannotated rt-MRI frames used for semi-supervised learning
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
./inference/videos
./inference/masks
./inference/videos_with_masks
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
