# DFEDet
Detail Fusion Enhancement Detector Network
## 1. Environment

### 1.1 Hardware

The experiments in the manuscript were conducted on:

- **VisDrone2019 / AI-TODv2:** `1 × NVIDIA RTX 4090`
- **SODA-D / SODA-A:** `4 × NVIDIA RTX 4090`

### 1.2 Core Frameworks

This project is implemented with:

- **PyTorch 1.10.0**  
- **CUDA  11.3**
- **MMDetection  2.13.0**
- **MMRotate**

## 2. Installation

We recommend using **Conda** to create an isolated environment.

### 2.1 Create environment

```bash
conda create -n dfedet python=3.8 -y
conda activate dfedet
```

### 2.2 Install PyTorch

Please install the exact PyTorch/CUDA build used in the manuscript.

```bash
# Example: install the PyTorch build consistent with the paper environment
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113
```

### 2.3 Install OpenMMLab dependencies

```bash
pip install -U openmim
mim install mmcv-full==1.4.0
pip install mmdet==2.13.0
```

> If your final release uses separate dependency versions for `mmdetection/` and `sodaa/`, please provide two independent conda environments or a compatibility table.

### 2.4 Clone this repository

```bash
git clone https://github.com/AuroraLove/DFEDet.git
cd DFEDet
```

### 2.5 Install the `mmdetection/` branch

```bash
cd mmdetection
pip install -r requirements.txt
pip install -v -e .
cd ..
```

### 2.6 Install the `sodaa/` branch

```bash
cd sodaa
pip install -r requirements.txt
pip install -v -e .
cd ..
```

### 2.7 Verify installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import mmdet; print(mmdet.__version__)"
python -c "import mmrotate; print(mmrotate.__version__)"
```

------

## 3. Dataset Preparation

The experiments are conducted on four public datasets:

- **VisDrone2019**
- **AI-TODv2**
- **SODA-D**
- **SODA-A**

Please download the datasets from their **official websites or official release pages**.
 After downloading, organize the image files and annotations into the expected directory structure used by this repository.

For training and evaluation, the annotations should be converted or organized into **COCO-style format**.
 For rotated detection tasks (e.g., **SODA-A**), please additionally ensure that the annotation format is compatible with the corresponding **MMRotate** data pipeline.

## 4. Training

## 4.1 VisDrone2019 / AI-TODv2

```bash
cd mmdetection

# VisDrone2019
python tools/train.py \
    configs/dfedet/dfe_det_visdrone_1x.py \
    --work-dir work_dirs/dfe_det_visdrone

# AI-TODv2
python tools/train.py \
    configs/dfedet/dfe_det_aitodv2_1x.py \
    --work-dir work_dirs/dfe_det_aitodv2
```

If distributed training is used:

```bash
bash tools/dist_train.sh \
    configs/dfedet/[config_file.py] \
    [num_gpus]
```

## 4.2 SODA-D

```bash
cd mmdetection

bash tools/dist_train.sh \
    configs/dfedet/dfe_det_sodad_1x.py \
    4
```

## 4.3 SODA-A

```bash
cd sodaa

bash tools/dist_train.sh \
    configs/dfe_det_sodaa/dfe_det_rcnn_r50_1x.py \
    4
```

------

## 5. Citation

If you use this repository, please cite the corresponding manuscript and the archived code/data release.

```bibtex

```
