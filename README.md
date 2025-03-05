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
