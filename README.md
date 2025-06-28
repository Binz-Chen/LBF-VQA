# LBF-VQA: Towards Language Bias-Free Visual Question Answering with Multi-Space Collaborative Debiasing
Code release for "LBF-VQA: Towards Language Bias-Free Visual Question Answering with Multi-Space Collaborative Debiasing". LBF-VQA can extend to other datasets with language biases.

# Requirements
* python 3.7.11
* pytorch 1.10.2+cu113
* torchvision 0.11.3+cu113

# Installation
```bash
git clone https://github.com/Binz-Chen/LBF-VQA
conda create -n LBF-VQA python=3.7
conda activate LBF-VQA
pip install -r requirements.txt
```

## Data Setup
- Almost all flags can be set at `utils/config.py`. The dataset paths, the hyperparams can be set accordingly in this file
- Download the VQA-CP datasets by executing `bash tools/download.sh`
- Download image features by following instructions from : https://github.com/hengyuan-hu/bottom-up-attention-vqa or https://github.com/airsplay/lxmert into `/data/vqacp-v2/detection_features` folder
- Preprocess process the data with `bash tools/process.sh`
- The pre-trained Glove features can be accessed via https://nlp.stanford.edu/projects/glove/

After downloading the datasets, keep them in the folders set by `utils/config.py`.
The new constructed dataset `SLAKE-LB` is provided from : https://drive.google.com/drive/folders/1j5dMeTCdiFbvZfyfjp6uRz-ldR5HvaPh?usp=drive_link.

## Preprocessing

The preprocessing steps are as follows:

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```

2. process answers and question types, and generate the frequency-based margins:
    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```

## Model training instruction
```
    python main_arcface.py --name test-VQA --gpu 0 --dataset DATASET
   ```
Set `DATASET` to a specfic dataset such as `SLAKE`, `SLAKE-LB`, `VQA-v2`, and `VQACP-v2`. 

## Model evaluation instruction
```
    python main_arcface.py --name test-VQA --test
   ```
