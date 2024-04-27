# MTMSD: Exploring Multi-Timestep Multi-Stage Diffusion Features for Hyperspectral Image Classification

This repository is the official implementation of MTMSD: Exploring Multi-Timestep Multi-Stage Diffusion Features for Hyperspectral Image Classification 

## Requirements
Please follow the instructions below to install the required packages. We run the code and obtain the results on PyTorch 2.0.1 and CUDA 11.7. 
```
conda create -n MTMSD python=3.9
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge mpi4py mpich
pip install scikit-learn blobfile scipy tqdm
```

## Datasets
Download following datasets:
* Indian Pines
* Pavia University
* Houston 2018
* WHU-Hi-LongKou
  
Then organize these datasets like:
```
datasets
   Indian_Pines
      Indian_pines_corrected.mat
      Indian_pines_gt.mat
   PaviaU
      PaviaU.mat
      PaviaU_gt.mat
   Houston2018
      Houston2018.mat
      Houston2018_gt.mat
   WHU-Hi-LongKou
      WHU_Hi_LongKou.mat
      WHU_Hi_LongKou_gt.mat
```

## How to use it?
1. Prepare diffusion features. The diffusion features extracted from the unsupervisedly pretrained ddpm model are needed for the finetuning. We provide two ways to get these features.
   
   * Option 1: Download the pretrained diffusion weights and extract features for each dataset.

     The pretrained weights are available at:
     BaiDuNetdisk：https://pan.baidu.com/s/1gRGBTUdB7GrChFlv46lZDQ?pwd=v2tn 
     Extraction Code：v2tn

     After downloading the pretrained weights and putting them under the /checkpoints directory, run the following commands to extract features for each dataset:
     ```
     bash scripts/feature_extract_ip.sh
     ```
     ```
     bash scripts/feature_extract_pu.sh
     ```
     ```
     bash scripts/feature_extract_hu18.sh
     ```
     ```
     bash scripts/feature_extract_longkou.sh
     ```
   
   * Option 2: Download and unzip the extracted features directly and put them under the /saved_features directory.

     The extracted features are available at:
     BaiDuNetdisk： https://pan.baidu.com/s/1zajMlZQtR6KGdZtc-sIN_g?pwd=1ahi
     Extraction Code：1ahi

2. To finetune the classifiers on the four datasets, run

   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/train_test_ip.sh
   ```
   
   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/train_test_pu.sh
   ```
   
   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/train_test_hu18.sh
   ```
   
   ```
   CUDA_VISIBLE_DEVICES=0 bash scripts/train_test_longkou.sh
   ```
