# Self-Supervised Learning for Knee Injury Diagnosis from Magnetic Resonance Data

This code repository contains code in Jupyter Notebook for the above mentioned paper in PyTorch

The pretraining_downstream.ipynb file is for training the model on one of the planes and subsequent donwstream classification on MRNet or KneeMRI dataset.

The ensembling.ipynb notebook is for obtaining the final predictions for all the three planes for MRNet dataset only. Since KneeMRI dataset contains only one plane (Sagittal), ensembling is not required when using KneeMRI dataset.

## Instructions for evaluating the SKID on the MRNet dataset

* MRNet dataset is copyright protected. For downloading the MRNet dataset, go to https://stanfordmlgroup.github.io/competitions/mrnet/
* Download and extract the dataset to a folder named __'MRNet-v1.0'__.
* Create a directory named __'models'__ and within that directory create two folders __'pretext'__ and __'downstream'__.
* The pretext and downstream models are uploaded on Google Drive. 
* For downloading the pretext models, go to https://drive.google.com/drive/folders/16FhskmQS-DXsVLacrISmpKcM5U_qcMNe?usp=sharing
* For downloading the downstream models, go to https://drive.google.com/drive/folders/1DW7_cju-ILrJUU_iL-z_gVPVYqwxakvY?usp=sharing
* Download the models into respective folders.
* To evaluate the models on the Validation set, run `python3 evaluate.py /path/to/MRNet-v1.0/ /path/to/models/`
