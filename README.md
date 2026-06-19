# Larynx segmentation for mid-sagittal speech production real-time tpeech MRI

This repository contains data, checkpoints, and code for larynx segmentation in mid-sagittal real-time speech production MRI.

---

## Dataset
The dataset consists of JPG-format MRI frames and corresponding JSON-format segmentation annotations.

---

## Environment Setup

```bash
chmod +x setup_env.sh
./setup_env.sh
```

The setup script installs required dependencies, including Detectron2.

---

## Training

Train Mask2Former models using the annotated data:

```bash
python scripts/train_maskformer.py
```

Train teacher-student semi-supervised learning models:

```bash
python scripts/train_ssl.py
```

---
## Inference
Pretrained checkpoints for both supervised and semi-supervised models trained under 25% and 100% labeled-data conditions are available in:
Sample videos are avaiable in: 
Generate segmentation masks for new MRI videos:

```bash
python inference.py
```

---
## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{zhang2026larynx,
  author    = {Zhang, Yubin and others},
  title     = {Larynx segmentation in mid-sagittal speech production real-Time MRI},
  booktitle = {Proceedings of Interspeech},
  year      = {2026}
}
```

---

## License

This repository is released for research and educational purposes.
