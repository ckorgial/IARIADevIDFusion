# Camera Model Identification Using Audio and Visual Content from Videos

This repository contains the code for the paper *Camera Model Identification Using Audio and Visual Content from Videos* introduced
by Ioannis Tsingalis, Christos Korgialas and Constantine Kotropoulos

## Abstract

The identification of device brands and models plays a pivotal role in the realm of multimedia forensic applications. This paper presents a framework capable of identifying devices using audio, visual content, or a fusion of them. The fusion of visual and audio content occurs in a later stage through the application of two fundamental fusion rules, namely the product and the sum rule. The device identification problem is tackled as a classification one by leveraging Convolutional Neural Networks. Experimental evaluation illustrates that the proposed framework exhibits promising classification performance when using audio or visual content independently. Furthermore, although the fusion results don't consistently surpass both individual modalities, they demonstrate promising potential for enhancing classification performance. Future research could concentrate on refining the fusion process to consistently improve classification performance in both modalities. Finally, a statistical significance test is performed for a more in-depth study of the overall classification results. 

----------------------------------------------------------
## Usage
### 1. Repository preparation 
Clone the repository, extract the files from [here](https://drive.google.com/file/d/1_JKC1RAkp-sL2rr5DfUb2-pVHW2mBHUl/view?usp=sharing) and [here](https://drive.google.com/file/d/1a6mv3HCyl4cvhFuOXlWgkiyXHQIdFl3w/view?usp=sharing) in the folders `/IARIADevIDFusion/image/results/` and `/IARIADevIDFusion/audio/results/`, respectively. The folders should have the structure 

```angular2html
├── results
│   ├── Native
│   │   ├── fold0
│   │   │   └── run1
│   │   │       ├── args.json
│   │   │       ├── logs
│   │   │       ├── model_best.ckpt
│   │   │       ├── model.ckpt
│   │   │       ├── proba.pkl
│   │   │       ├── tr_acc.log
│   │   │       ├── tr_loss.log
│   │   │       ├── val_acc.log
│   │   │       └── val_loss.log
│   │   ├── fold1
│   │   │   └── run1
│   │   │       ├── ...
│   │   ├── fold2
│   │   │   └── run1
│   │   │       ├── ...
│   │   ├── fold3
│   │   │   └── run1
│   │   │       ├── ...
│   │   └── fold4
│   │       └── run1
│   │           ├── ...
│   ├── WA
│   │   ├── ...
│   └── YT
│       ├── ...
```

### 2. Install requirements
The requirements are in the *requirements.txt* file. 


### 3. Download datasets
You can download the VISION dataset from [here](https://lesc.dinfo.unifi.it/VISION/) using the script 

`/IARIADevIDFusion/datasets/VISION/downloadVISION.py`

This script creates the folder `/IARIADevIDFusion/datasets/VISION/dataset/` with structure

```angular2html
├── D01_Samsung_GalaxyS3Mini
│   ├── images
│   │   ├── flat
│   │   ├── nat
│   │   ├── natFBH
│   │   ├── natFBL
│   │   └── natWA
│   └── videos
│       ├── flat
│       ├── flatWA
│       ├── flatYT
│       ├── indoor
│       ├── indoorWA
│       ├── indoorYT
│       ├── outdoor
│       ├── outdoorWA
│       └── outdoorYT
├── D02_Apple_iPhone4s
│   ├── images
│   │   ├── flat
│   │   ├── nat
│   │   ├── natFBH
│   │   ├── natFBL
│   │   └── natWA
│   └── videos
│       ├── flat
│       ├── flatWA
│       ├── flatYT
│       ├── indoor
│       ├── indoorWA
│       ├── indoorYT
│       ├── outdoor
│       ├── outdoorWA
│       └── outdoorYT
...
...
```

### 4. Extract audio (.wav) files from videos
Run the script 
```angular2
/IARIADevIDFusion/audio/extractWav.py
```
This script creates the folder `/IARIADevIDFusion/audio/extractedWav/` with structure

```angular2html
├── D01_V_flat_move_0001
│   └── D01_V_flat_move_0001.wav
├── D01_V_flat_move_0002
│   └── D01_V_flat_move_0002.wav
├── D01_V_flat_panrot_0001
│   └── D01_V_flat_panrot_0001.wav
├── D01_V_flat_still_0001
│   └── D01_V_flat_still_0001.wav
├── D01_V_flatWA_move_0001
│   └── D01_V_flatWA_move_0001.wav
├── D01_V_flatWA_move_0002
│   └── D01_V_flatWA_move_0002.wav
├── D01_V_flatWA_panrot_0001
│   └── D01_V_flatWA_panrot_0001.wav
├── D01_V_flatWA_still_0001
...
...
```

### 5. Extract log mel spectrograms from the audio (.wav) files

Run the script 
```angular2
/IARIADevIDFusion/audio/extractMel.py
```

This script creates the folder `/IARIADevIDFusion/audio/extractedMel/` with structure
```angular2html
├── D01_V_flat_move_0001
│   ├── D01_V_flat_move_0001_chanel0.png
│   ├── D01_V_flat_move_0001_chanel1.png
│   ├── D01_V_flat_move_0001_chanel2.png
│   └── D01_V_flat_move_0001.pkl
├── D01_V_flat_move_0002
│   ├── D01_V_flat_move_0002_chanel0.png
│   ├── D01_V_flat_move_0002_chanel1.png
│   ├── D01_V_flat_move_0002_chanel2.png
│   └── D01_V_flat_move_0002.pkl
├── D01_V_flat_panrot_0001
│   ├── D01_V_flat_panrot_0001_chanel0.png
│   ├── D01_V_flat_panrot_0001_chanel1.png
│   ├── D01_V_flat_panrot_0001_chanel2.png
│   └── D01_V_flat_panrot_0001.pkl
├── D01_V_flat_still_0001
│   ├── D01_V_flat_still_0001_chanel0.png
│   ├── D01_V_flat_still_0001_chanel1.png
│   ├── D01_V_flat_still_0001_chanel2.png
│   └── D01_V_flat_still_0001.pkl
├── D01_V_flatWA_move_0001
│   ├── D01_V_flatWA_move_0001_chanel0.png
│   ├── D01_V_flatWA_move_0001_chanel1.png
│   ├── D01_V_flatWA_move_0001_chanel2.png
│   └── D01_V_flatWA_move_0001.pkl
```

### 6. Create training, evaluation and test Splits (a.k.a. folds)

Run the script 
```angular2
/IARIADevIDFusion/splits/create_splits.py
```

This script creates the folder `IARIADevIDFusion/splits/JoI_splits/` with subfolders `Native`, `WA`, and `YT`.

### 7. Train audio network
Run the script 
```angular2
/IARIADevIDFusion/audio/train_audio.py --visual_content YT --n_fold 0 --model MobileNetV3Large --results_dir results --epochs 100 --lr 1e-4 --optimizer Adam --project_dir /IARIADevIDFusion/audio/ --split_dir /IARIADevIDFusion/splits/JoI_splits --mel_dir /IARIADevIDFusion/audio/extractedMel
```
This script creates the folder `IARIADevIDFusion/audio/results/` with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the following structure

```angular2html
└── run1
    ├── args.json
    ├── logs
    ├── model_best.ckpt
    ├── model.ckpt
    ├── proba.pkl
    ├── tr_acc.log
    ├── tr_loss.log
    ├── val_acc.log
    └── val_loss.log
```

### 8. Evaluate audio network
Run the script 
```angular2
/IARIADevIDFusion/audio/eval_audio.py --n_run 1 --results_dir /IARIADevIDFusion/audio/results/ --project_dir /IARIADevIDFusion/audio/ --mel_dir /IARIADevIDFusion/audio/extractedMel --visual_content YT --n_fold 0
```

This script uses the best model `model_best.ckpt` of each fold and produced the accuracy results. 

### 9. Extract video frames
Run the script 
```angular2
/IARIADevIDFusion/datasets/VISION/extract_frames.py
```
This script creates the folder `/IARIADevIDFusion/datasets/VISION/extractedFrames/` with structure

```angular2html
├── D01_V_flat_move_0001
├── D01_V_flat_move_0002
├── D01_V_flat_panrot_0001
├── D01_V_flat_still_0001
├── D01_V_flatWA_move_0001
├── D01_V_flatWA_move_0002
├── D01_V_flatWA_panrot_0001
├── D01_V_flatWA_still_0001
├── D01_V_indoor_move_0001
├── D01_V_indoor_move_0002
├── D01_V_indoor_panrot_0001
├── D01_V_indoor_panrot_0002
├── D01_V_indoor_still_0001
...
...
```

Each folder contains the video frames saved in `.jpg` format.

### 10. Train image network
Run the script 
```angular2
/IARIADevIDFusion/audio/train_image.py --visual_content Native --n_fold 0 --model ResNet50 --project_dir /IARIADevIDFusion/image/ --epochs 100 --lr 1e-4 --vision_frames_dir /IARIADevIDFusion/datasets/VISION/extractedFrames --optimizer Adam --results_dir /IARIADevIDFusion/image/results/
```

This script creates the folder `IARIADevIDFusion/image/results/` with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the same structure when training the audio network.

### 11. Evaluate image network
Run the script 
```angular2
/IARIADevIDFusion/image/eval_image.py --n_run 1 --visual_content Native --n_fold 0 --vision_frames_dir /IARIADevIDFusion/datasets/VISION/extractedFrames --project_dir /IARIADevIDFusion/image/ --results_dir /IARIADevIDFusion/image/results/ --split_dir /IARIADevIDFusion/splits/JoI_splits
```

This script uses the best model `model_best.ckpt` of each fold and produces `proba.pkl` which contains the classification probabilities of each video frame. These probabilities are used to obtain the accuracy results.

### 12. Late fusion
Run the script 

```angular2html
/IARIADevIDFusion/fusion/late_fusion.py --n_run_audio_dir /IARIADevIDFusion/audio/results/Native/fold0/run1/ --n_run_image_dir /IARIADevIDFusion/image/results/Native/fold0/run1/
```

**This script produces the results in Tables II and III.**


### 13. Significance test
Run the script 

```angular2html
/IARIADevIDFusion/significance_test.py --n_run_audio 1 --audio_project_dir /IARIADevIDFusion/audio --n_run_image 1 --image_project_dir /IARIADevIDFusion/image --n_fold 0 --visual_content YT
```
**This script produces the results in Tables II and III along with the results in Tables IV and V.**
