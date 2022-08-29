# SKID: Self-Supervised Learning for Knee Injury Diagnosis from MRI Data

This code repository contains code in Jupyter Notebook for the above mentioned paper in PyTorch

The pretraining_downstream.ipynb file is for training the model on one of the planes and subsequent donwstream classification on MRNet or KneeMRI dataset.

The ensembling.ipynb notebook is for obtaining the final predictions for all the three planes for MRNet dataset only. Since KneeMRI dataset contains only one plane (Sagittal), ensembling is not required when using KneeMRI dataset.
