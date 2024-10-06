# Online Scene CAD Recomposition via Autonomous Scanning (Siggraph Asia 2023)

## News

2023.9.30 The code will be released soon!

2024.6.21 Uploaded source code V1!

2024.10.6 Added the detailed document!

## TODO

Release extra generated datasets.

Release pre-trained models.

Refine and simplify code.

Merge all packages.

## 0. Tested Environment Setting

```bash
Ubuntu 20.04
g++ 11
cuda 11.7
```

## 1. Dataset Download and File Structure

First, you need to download ScanNet, ShapeNet and Scan2CAD datasets, and make the files just like:

```bash
ScanNet --> <path/to/your/dataset/folder>/ScanNet/scans/scene0000_00/
ShapeNet --> <path/to/your/dataset/folder>/ShapeNet/Core/ShapeNetCore.v2/
Scan2CAD --> <path/to/your/dataset/folder>/Scan2CAD/scan2cad_dataset/object_position_dataset/
```

## 2. Conda Env Setup

We provide a simple setup bash file to manage everything! Now you can just run these commands:

```bash
conda create acr python=3.8
conda activate acr
./setup.sh
```

## 3. Extra Datasets Generation

We use some extra generated datasets to train our used models, and you can get all of them by running single script!

All you need to do is just updating all used dataset folder paths in the demo functions called by this script!

```bash
python generate_datasets.py
```

And this script will create these datasets:

```bash
<path/to/your/dataset/folder>/
  |- ScanNet/
      |- objects/
      |- bboxes/
      |- glb/
  |- Scan2CAD/
      |- object_model_maps/
  |- ShapeNet/
      |- udfs/
```

We are also uploading pre-processed datasets, and they will be available at BaiduNetDisk as soon as possible:

```bash
https://pan.baidu.com/s/1e1w2xtom4izmpyHn-a6zdg?pwd=chli
```

## 4. Train

Now, you have collected all used datasets, and it's time to start training kernel models!

## 4.1. Rotation

```bash
cd ./points-shape-detect/
python train_rotate.py
```

## 4.2. ABB detection

```bash
cd ./points-shape-detect/
python train.py
```

## 4.3. Pose Refinement

```bash
cd ./global-pose-refine/
python train.py
```

## 5. Run

We provide a simple running bash file to start a demo scanning process! Now you can just run this command to test your environment and start your trip:

```bash
./run.sh
```

## Citation

```bash
@article{Li-2023-Online,
  title = {Online Scene CAD Recomposition via Autonomous Scanning},
  author = {Changhao Li, Junfu Guo, Ruizhen Hu, Ligang Liu},
  journal = {ACM Transactions on Graphics (SIGGRAPH Asia 2023)},
  volume = {42},
  number = {6},
  pages = {Article 250: 1-16},
  year = {2023}}
```

## Enjoy it~
