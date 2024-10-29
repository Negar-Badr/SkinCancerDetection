# MAIS202_Project

## Overview

This project builds a web app to classify skin diseases based on uploaded images and user-provided metadata (e.g., onset, ethnicity). The goal is to provide an accurate and accessible diagnostic tool, leveraging deep learning for reliable predictions.

## Objectives

- Model Output: Skin disease classification.
- Deliverable: A user-friendly web app.

## Dataset

Source: 
- Skin Cancer MNIST: HAM10000 dataset on Kaggle (https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
Download:
- pip install kaggle
- kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
Note: Do not push dataset files to the repository.

## Workflow

Set up the development environment : TensorFlow or PyTorch for building deep learning models
Preprocessing: Augment and encode data; split into train, validation, and test sets. (Split data into 70% training, 20% validation, and 10% testing sets.)
Model: Fine-tune pre-trained models (e.g., ResNet or VGG) with added custom layers; train with metrics like accuracy and ROC-AUC.
Evaluation: Test on holdout data; refine with metrics and tuning if needed.
Web App: Develop with Flask for image upload and prediction display and deploy it.

