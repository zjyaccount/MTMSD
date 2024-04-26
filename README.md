# MTMSD:Exploring Multi-Timestep Multi-Stage Diffusion Features for Hyperspectral Image Classification
It takes some time for the features to be uploaded. They will be updated in a few days.

# Requirements
The code has been tested with PyTorch 2.0 and Cuda 11.7

# How to use it?
1. Prepare datasets and features. The features extracted from the unsupervisedly pretrained ddpm model are needed for the finetuning. We provide two ways to get these features.
   
   a. Download the pretrained weights and extract features for each dataset.
      BaiDuNetdisk：https://pan.baidu.com/s/1gRGBTUdB7GrChFlv46lZDQ?pwd=v2tn 
      Extraction Code：v2tn
   
   b. Download the extracted features directly and put them in the /saved_features

2. To finetune the classifiers on the four datasets, run

'''
CUDA_VISIBLE_DEVICES=0 bash scripts/train_test_ip.sh
'''
