# Explainable Camera Model Identification Employing log-Mel Spectrograms from Videos’ Audio

This repository contains the code for the paper *Explainable Camera Model Identification Employing log-Mel Spectrograms from Videos’ Audio* introduced by Christos Korgialas, Ioannis Tsingalis, and Constantine Kotropoulos

----------------------------------------------------------
## Usage
### 1. Repository preparation 
Clone the repository, extract the files from [here](https://drive.google.com/file/d/1_JKC1RAkp-sL2rr5DfUb2-pVHW2mBHUl/view?usp=sharing) and [here](https://drive.google.com/file/d/1a6mv3HCyl4cvhFuOXlWgkiyXHQIdFl3w/view?usp=sharing) in the folders `/XAIDevIDFusion/image/results/` and `/XAIDevIDFusion/audio/results/`, respectively. The folders should have the structure 

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

`/XAIDevIDFusion/datasets/VISION/downloadVISION.py`

This script creates the folder `/XAIDevIDFusion/datasets/VISION/dataset/` with structure

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
/XAIDevIDFusion/audio/extractWav.py
```

This script creates the folder `/XAIDevIDFusion/audio/extractedWav/` with structure

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
/XAIDevIDFusion/audio/extractMel.py
```
or for the band-pass filtered data 

```angular2
/XAIDevIDFusion/audio/extractMel_filtered.py
```

This script creates the folder `/XAIDevIDFusion/audio/extractedMel/` with structure
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
/XAIDevIDFusion/splits/create_splits.py
```

This script creates the folder `XAIDevIDFusion/splits/JoI_splits/` with subfolders `Native`, `WA`, and `YT`.

### 7. Grad-CAM Heatmap Generation

Run the script 
```angular2
/XAIDevIDFusion/gradcam_fusion.py
```

### 8. Train audio network
Run the script 
```angular2
/XAIDevIDFusion/audio/train_audio.py --visual_content YT --n_fold 0 --model MobileNetV3Large --results_dir results --epochs 100 --lr 1e-4 --optimizer Adam --project_dir /XAIDevIDFusion/audio/ --split_dir /XAIDevIDFusion/splits/JoI_splits --mel_dir /XAIDevIDFusion/audio/extractedMel
```
This script creates the folder `XAIDevIDFusion/audio/results/` with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i=0, 1, 2, 3, 4`. Each fold folder has the following structure

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

### 9. Evaluate audio network
Run the script 
```angular2
/XAIDevIDFusion/audio/eval_audio.py --n_run 1 --results_dir /XAIDevIDFusion/audio/results/ --project_dir /XAIDevIDFusion/audio/ --mel_dir /XAIDevIDFusion/audio/extractedMel --visual_content YT --n_fold 0
```

This script uses the best model `model_best.ckpt` of each fold and produced the accuracy results. 

### 10. Extract video frames
Run the script 
```angular2
/XAIDevIDFusion/datasets/VISION/extract_frames.py
```
This script creates the folder `/XAIDevIDFusion/datasets/VISION/extractedFrames/` with structure

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

### 11. Train image network
Run the script 
```angular2
/XAIDevIDFusion/audio/train_image.py --visual_content Native --n_fold 0 --model ResNet50 --project_dir /XAIDevIDFusion/image/ --epochs 100 --lr 1e-4 --vision_frames_dir /XAIDevIDFusion/datasets/VISION/extractedFrames --optimizer Adam --results_dir /XAIDevIDFusion/image/results/
```

This script creates the folder `XAIDevIDFusion/image/results/` with subfolders `Native`, `WA`, and `YT`, where the best model `model_best.ckpt` is saved for each `fold{i}`, with `i = 0, 1, 2, 3, 4`. Each fold folder has the same structure when training the audio network.

### 12. Evaluate image network
Run the script 
```angular2
/XAIDevIDFusion/image/eval_image.py --n_run 1 --visual_content Native --n_fold 0 --vision_frames_dir /XAIDevIDFusion/datasets/VISION/extractedFrames --project_dir /XAIDevIDFusion/image/ --results_dir /XAIDevIDFusion/image/results/ --split_dir /XAIDevIDFusion/splits/JoI_splits
```

This script uses the best model `model_best.ckpt` of each fold and produces `proba.pkl` which contains the classification probabilities of each video frame. These probabilities are used to obtain the accuracy results.

### 13. Late fusion
Run the script 

```angular2html
/XAIDevIDFusion/fusion/late_fusion.py --n_run_audio_dir /XAIDevIDFusion/audio/results/Native/fold0/run1/ --n_run_image_dir /XAIDevIDFusion/image/results/Native/fold0/run1/
```

**This script produces the results in Tables II and III.**


### 14. Significance test
Run the script 

```angular2html
/XAIDevIDFusion/significance_test.py --n_run_audio 1 --audio_project_dir /XAIDevIDFusion/audio --n_run_image 1 --image_project_dir /XAIDevIDFusion/image --n_fold 0 --visual_content YT
```
**This script produces the results in Tables II and III along with the results in Tables IV and V.**


## Reference
If you use this code in your experiments please cite this work by using the following bibtex entry:

```
@inproceedings{tobeUpdated,
  title={Explainable Camera Model Identification Employing log-Mel Spectrograms from Videos’ Audio},
  author={Korgialas, Christos and Tsingalis, Ioannis and Kotropoulos, Constantine},
  booktitle={Asilomar Conference on Signals, Systems, and Computers},
  pages={1--5},
  year={2024}
}
```
