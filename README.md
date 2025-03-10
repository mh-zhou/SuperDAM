# To be CONTINUE
- More information, you can contact us: sjtutsb@sjtu.edu.cn, {mhzhou0412; tonxycs}@gmail.com[Preferred]
  
## Core Part:
![DGM](https://github.com/user-attachments/assets/9b20e826-1586-497e-9be5-455398bef0c7)
Notes： Dynamic Grouping Mechanism (DGM), the core of SuperDAM, centralizes DCL by generating, weighting, and adjusting dynamic features. These features are then processed through grouped convolution to extract critical channel-wise information.



![DCL](https://github.com/user-attachments/assets/83d81825-eb62-4375-8bd7-c87eb3756a3b)
Notes: ##
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
# SuperDAM Dateset-split(the part is necessary for your beginning)
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

