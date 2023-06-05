# **Three Method Fine-tune on CLIP for Vehicle Counting Task**

## **Approach**
### 1. General
### 2. Adapter
### 3. VPT (shallow)
<br/>

---
<br/>

## **Hardware**

* ### CPU: AMD EPYC 7742 64-Core Processor
* ### RAM: 512GB
* ### GPU: Nvidia A100 (40GB VRAM)
* ### Disk Space Available: 1TB
<br/>

---
<br/>

## **Install the Required Packages**

```bash
$ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
$ pip install ftfy regex tqdm torch-summary
$ pip install git+https://github.com/openai/CLIP.git
```
<br/>

---
<br/>

## **Prepare KITTI Dataset**
> Dataset:  
[Dataset Link](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)  
*Note: only need left color images of object data set (12 GB) and training labels of object data set (5 MB).
``` python
# And you must organize files into the following structure:

kitti_dataset
     ├── testing
     |      └── image_2 #Only including testing img files
     └── training
            ├── image_2 #Only including training img files
            └── label_2 #Only including txt files
```
<br/>

---
<br/>

## **Preprocessing Label**
```bash
# You should modify the path of your training image_2 folder by yourself in the script (Line 4: kitti_label_file_path).
python text_generation.py
```
<br/>

---
<br/>

## **Fine-tune**
```bash
# Replace "../KITTI_DATASET_ROOT/training/image_2/" into the path of your training image_2 folder.

# General fine tune on whole model
python train.py --kitti_image_file_path "../KITTI_DATASET_ROOT/training/image_2/"

# Using adapter to fine tune
python train.py --adapter --kitti_image_file_path "../KITTI_DATASET_ROOT/training/image_2/"


# Using vpt to fine tune
python train.py --prompt --vpt_version 1or2 --kitti_image_file_path "../KITTI_DATASET_ROOT/training/image_2/"
```
<br/>

---
<br/>

## **test**
```bash
# Replace "../KITTI_DATASET_ROOT/training/image_2/" into the path of your training image_2 folder.

# General fine tune on whole model
python test.py --kitti_image_file_path "../KITTI_DATASET_ROOT/training/image_2/"

# Using adapter to fine tune
python test.py --adapter --kitti_image_file_path "../KITTI_DATASET_ROOT/training/image_2/"


# Using vpt to fine tune
python test.py --prompt --vpt_version 1or2 --kitti_image_file_path "../KITTI_DATASET_ROOT/training/image_2/"
```
<br/>

---
<br/>

## **Acknowledgement**
This repo benefits from [CLIP](https://github.com/openai/CLIP), [AIM](https://github.com/taoyang1122/adapt-image-models), and [VPT](https://github.com/KMnP/vpt). Thanks for their wonderful works.
