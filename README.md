# MAIS202 Project: Skin Disease Classification Web App

## Overview

This project aims to develop a web application that can classify if a mole is skin cancer or not based on an uploaded image and metadata, such as the gender, the age, and other relevant details. The application leverages deep learning to provide accurate, user-friendly diagnostics, potentially benefiting telemedicine and dermatology diagnostics by giving a first diagnosis to the patient. The webapp will give a percentage of accuracy and remember the person to consult a doctor.

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

- Set up the development environment : TensorFlow or PyTorch for building deep learning models
- Preprocessing: Augment and encode data; split into train, validation, and test sets. (Split data into 70% training, 20% validation, and 10% testing sets.)
- Model: Fine-tune pre-trained models (e.g., ResNet or VGG) with added custom layers; train with metrics like accuracy and ROC-AUC.
- Evaluation: Test on holdout data; refine with metrics and tuning if needed.
- Web App: Develop with Flask for image upload and prediction display and deploy it.

