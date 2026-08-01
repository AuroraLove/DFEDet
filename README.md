# DFEDet: Semantic-Guided Directional Wavelet Detail Reconstruction for Small Object Detection

This repository provides the implementation and experiment configurations for **DFEDet**, a frequency-domain feature-enhancement framework for small object detection. DFEDet uses low-frequency semantic guidance to identify object-relevant regions and reconstructs directional high-frequency details in shallow high-resolution features.

## 1. Repository Structure

```text
DFEDet/
├── mmdetection/   # VisDrone2019, AI-TODv2, and SODA-D
├── sodaa/         # SODA-A oriented-object detection
├── README.md
└── LICENSE
```

The benchmark-specific configuration files are:

```text
mmdetection/configs/dfedet/dfe_det_visdrone_1x.py
mmdetection/configs/dfedet/dfe_det_aitodv2_1x.py
mmdetection/configs/dfedet/dfe_det_sodad_1x.py
sodaa/configs/dfe_det_sodaa/dfe_det_rcnn_r50_1x.py
```

## 2. Environment

### 2.1 Hardware Used in the Manuscript

- **VisDrone2019 / AI-TODv2:** `1 × NVIDIA RTX 4090`
- **SODA-D / SODA-A:** `2 × NVIDIA RTX 3090`

Equivalent GPUs can be used, but batch size, learning rate, and runtime may need to be adjusted accordingly.

### 2.2 Software Versions

The released code is based on the following environment:

- Python `3.8`
- PyTorch `1.10.0`
- torchvision `0.11.1`
- CUDA `11.3`
- MMCV-Full `1.6.0`
- MMDetection `2.26.0`
- MMRotate `0.3.3`

## 3. Installation

We recommend using Conda to create an isolated environment.

### 3.1 Create the environment

```bash
conda create -n dfedet python=3.8 -y
conda activate dfedet
```

### 3.2 Install PyTorch and torchvision

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### 3.3 Install MMCV

```bash
pip install -U openmim
mim install "mmcv-full==1.6.0"
```

### 3.4 Clone the repository

```bash
git clone https://github.com/AuroraLove/DFEDet.git
cd DFEDet
```

### 3.5 Install the MMDetection-based code

```bash
cd mmdetection
pip install -r requirements.txt
pip install -v -e .
cd ..
```

### 3.6 Install the MMRotate-based code

```bash
cd sodaa
pip install -r requirements.txt
pip install -v -e .
cd ..
```

### 3.7 Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import mmcv; print('MMCV:', mmcv.__version__)"
python -c "import mmdet; print('MMDetection:', mmdet.__version__)"
python -c "import mmrotate; print('MMRotate:', mmrotate.__version__)"
```

Expected core versions are:

```text
PyTorch: 1.10.0
MMCV: 1.6.0
MMDetection: 2.26.0
MMRotate: 0.3.3
```

## 4. Dataset Preparation

Experiments are conducted on four public benchmarks:

- VisDrone2019
- AI-TODv2
- SODA-D
- SODA-A

The datasets are **not redistributed** in this repository. Please download them from their official websites or official release pages and comply with their respective licenses and terms of use.

After downloading each dataset, update the corresponding `data_root`, image paths, and annotation paths in the dataset configuration files:

```text
mmdetection/configs/_base_/datasets/visdrone.py
mmdetection/configs/_base_/datasets/aitodv2.py
mmdetection/configs/_base_/datasets/sodad.py
sodaa/configs/_base_/datasets/sodaa.py
```

### 4.1 VisDrone2019 and AI-TODv2

The MMDetection branch expects COCO-style annotations. Organize the converted datasets according to the paths specified in the corresponding dataset configuration files.

A typical structure is:

```text
data/
├── VisDrone2019/
│   ├── train2017/
│   ├── val2017/
│   └── annotations/
└── AI-TODv2/
    ├── train2017/
    ├── val2017/
    └── annotations/
```

The exact folder names may be changed by editing the corresponding dataset configuration.

### 4.2 SODA-D

The repository provides SODA-D preprocessing utilities under:

```text
mmdetection/tools/img_split/
```

Before running the split script, edit the paths and split parameters in:

```text
mmdetection/tools/img_split/split_configs/
```

To match the manuscript protocol, use:

- crop size: `800 × 800`
- stride: `650`
- network input after cropping: `1200 × 1200`

Example:

```bash
cd mmdetection
python tools/img_split/sodad_split.py \
    --cfgJson tools/img_split/split_configs/split_train.json
```

Repeat the procedure for validation and test splits. The repository also includes `generate_wo_ignore.py` and the corresponding evaluation utilities. Evaluation follows the official SODA protocol on the original-image coordinate system after patch predictions are mapped back and merged.

### 4.3 SODA-A

The repository provides SODA-A preprocessing utilities under:

```text
sodaa/tools/data/sodaa/
```

Before running the split script, edit the image, annotation, output, and split settings in the corresponding JSON files. To match the manuscript protocol, use:

- crop size: `800 × 800`
- stride: `650`
- network input after cropping: `1200 × 1200`
- oriented-box convention: `le90`

Example:

```bash
cd sodaa
python tools/data/sodaa/sodaa_split.py \
    --base-json sodaa_train.json
```

Repeat the procedure for validation and test splits. The provided SODA-A utilities include ignored-region filtering and the official original-image evaluation workflow.

## 5. Training

All controlled ablation experiments use random seed `42`. The commands below also enable deterministic CuDNN behavior for improved repeatability.

### 5.1 VisDrone2019

```bash
cd mmdetection
python tools/train.py \
    configs/dfedet/dfe_det_visdrone_1x.py \
    --work-dir work_dirs/dfedet_visdrone \
    --seed 42 \
    --deterministic
```

### 5.2 AI-TODv2

```bash
cd mmdetection
python tools/train.py \
    configs/dfedet/dfe_det_aitodv2_1x.py \
    --work-dir work_dirs/dfedet_aitodv2 \
    --seed 42 \
    --deterministic
```

### 5.3 SODA-D

```bash
cd mmdetection
bash tools/dist_train.sh \
    configs/dfedet/dfe_det_sodad_1x.py \
    2 \
    --work-dir work_dirs/dfedet_sodad \
    --seed 42 \
    --deterministic
```

### 5.4 SODA-A

```bash
cd sodaa
bash tools/dist_train.sh \
    configs/dfe_det_sodaa/dfe_det_rcnn_r50_1x.py \
    2 \
    --work-dir work_dirs/dfedet_sodaa \
    --seed 42 \
    --deterministic
```

Training logs, resolved configuration files, environment information, random seeds, and checkpoints are automatically written to the specified `work_dirs/` directory.

## 6. Evaluation

Replace `<checkpoint.pth>` with the checkpoint to be evaluated.

### 6.1 VisDrone2019

```bash
cd mmdetection
python tools/test.py \
    configs/dfedet/dfe_det_visdrone_1x.py \
    <checkpoint.pth> \
    --eval bbox
```

### 6.2 AI-TODv2

```bash
cd mmdetection
python tools/test.py \
    configs/dfedet/dfe_det_aitodv2_1x.py \
    <checkpoint.pth> \
    --eval bbox
```

### 6.3 SODA-D

```bash
cd mmdetection
python tools/test.py \
    configs/dfedet/dfe_det_sodad_1x.py \
    <checkpoint.pth> \
    --eval bbox
```

### 6.4 SODA-A

```bash
cd sodaa
python tools/test.py \
    configs/dfe_det_sodaa/dfe_det_rcnn_r50_1x.py \
    <checkpoint.pth> \
    --eval mAP
```

For SODA-D and SODA-A, follow the official patch-to-original-image merging and evaluation procedures included in the corresponding data utilities.

## 7. Expected Results

The following values are the AP results reported in the manuscript under the stated data-processing and training protocols.

| Dataset | Configuration | Reported AP |
|---|---|---:|
| VisDrone2019 | `dfe_det_visdrone_1x.py` | 30.9 |
| AI-TODv2 | `dfe_det_aitodv2_1x.py` | 26.3 |
| SODA-D | `dfe_det_sodad_1x.py` | 33.2 |
| SODA-A | `dfe_det_rcnn_r50_1x.py` | 36.4 |

Minor numerical differences may occur because of CUDA kernels, GPU models, dependency builds, and nondeterministic low-level operations.

## 8. Reproducibility Checklist

- [x] Complete DFEDet source code
- [x] Benchmark-specific configuration files
- [x] Exact core package versions
- [x] Dataset preparation instructions
- [x] SODA-D cropping utilities
- [x] SODA-A cropping utilities
- [x] Training commands for all four benchmarks
- [x] Evaluation commands for all four benchmarks
- [x] Fixed random seed for controlled ablations
- [x] Hardware information
- [x] Expected AP values
- [x] Automatic experiment logging through `work_dirs/`
- [x] Data and code availability statement
- [x] License information

## 9. Data and Code Availability

The complete DFEDet implementation, benchmark-specific configurations, preprocessing utilities, and training/evaluation entry points are provided in this repository. The four evaluation datasets are publicly available and must be obtained from their official distribution pages. Dataset files are not redistributed here.

## 10. License

DFEDet-specific code and top-level documentation are released under the **MIT License**. The bundled MMDetection- and MMRotate-derived components retain their original **Apache License 2.0** notices and terms. The datasets are governed by their respective official licenses and terms of use.

## 11. Citation

Please cite the manuscript when using this repository:

```bibtex
@misc{yu2026dfedet,
  title  = {DFEDet: Semantic-Guided Directional Wavelet Detail Reconstruction for Small Object Detection},
  author = {Yu, Zheng and Guo, Li and Xu, Qing and Huo, Hongyuan and Ren, Fang and Cui, Qifan},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## 12. Acknowledgements

This repository is built upon MMDetection and MMRotate. We thank the OpenMMLab community and the authors of the public datasets and baseline methods used in this work.
