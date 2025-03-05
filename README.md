## Getting Started

### Dependency
Our released implementation is tested on(2D-Best&3D).
+ Python 3.9.19 
+ PyTorch 2.0.1
+ NVIDIA CUDA 11.8
+ NVIDIA Geforce RTX 4090 GPUs

We also tested on(2D&3D-Best).
+ Python 3.8.20 
+ PyTorch 1.12.1
+ NVIDIA CUDA 11.3
+ NVIDIA Geforce RTX 3090 GPUs
# SuperDAM Dateset-split
```
SuperDAM
├── 2d-data(jpg/png...)
│   ├── AD-MRI
│   │   │── 1/
│   │   │── MildDemented(*)
│   │   │── ModerateDemented(*)
│   │   │── NonDemented(*)
│   │   │   │── train/*
│   │   │   │── val/*
│   ├── BT-MRI
│   │   │── 1/
│   │   │── Glioma(*)
│   │   │── Healthy(*)
│   │   │── Meningioma(*)
│   │   │   │── train/*
│   │   │   │── val/*
│   ├── BT-CT
│   │   │── 1/
│   │   │── Healthy(*)
│   │   │── Tumor(*)
│   │   │── TuomrBig(*)
│   │   │   │── train/*
│   │   │   │── val/*
```

```
SuperDAM
├── 3d-data(.nii)
│   ├── AD-MRI
│   │   │── 3dAD
│   │   │── 3dCN
│   │   │── Preprocessed/
│   │   │   │── AD/.nii/
│   │   │   │── CN/.nii/
│   │   │   │── ~fold_CNvsAD_0/1/2/3/4.csv
│   │   │   │── ~combined_train_list_0/1/2.csv
```
