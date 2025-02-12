# Face Recognition Using a Siamese Network (forsn)

## Overview

This project implements a **face recognition system** using a **Siamese network** for **one-shot learning**. The goal is to improve the accuracy of an existing model used by the **H-BRS @home lab’s Human Support Robot (HSR)**, which tracks and identifies individuals entering the lab.

## Problem Statement

The previous face recognition model suffered from **misidentification issues** and **overfitting**, showing **100% accuracy on training data** but failing on unseen test data. The task was to either **modify the existing model** or develop a **new one** to address these challenges.

## Dataset

- **Training Data**: MUCT face database (3,755 images with diverse lighting and ethnicity)
- **Testing Data**: Custom dataset created with images taken in the lab (186x186 resolution)
- **Data Augmentation**: Applied techniques like shadow effects, color variations, and shear transformations to enhance generalization.

## Model Architecture

A **Siamese network** is used, consisting of:

### **Convolutional Neural Network**
- Conv2D(3, 32, 5), PReLU()
- MaxPool2D(2, stride=2)
- Conv2D(32, 64, 5), PReLU()
- MaxPool2D(2, stride=2)

### **Fully Connected Network**
- Linear(30976, 64), PReLU()
- Linear(64, 64), PReLU()
- Linear(64, 50)

### **Loss Function**
- **Triplet Loss** with margin = 0.1

### **Optimizer**
- **Adam** with learning rate scheduling (step=5, gamma=0.1)

## Key Improvements

- **Weight Initialization (Kaiming)**: To prevent vanishing gradients.
- **Batch Normalization**: Added in both convolutional and fully connected layers.
- **Input Normalization**: Improved network stability.
- **Custom Testing Dataset**: Increased resolution and diversity.
- **KNN Classifier**: Enables face database updates without retraining.

## Evaluation & Results

| Batch Size | Learning Rate | WI | WI+BN | WI+BN+IN |
|------------|-------------|----|------|------|
| 8         | 0.1         | 0  | 0.339 | 0.661 |
| 8         | 0.01        | 0.793 | 0.509 | 0.661 |
| 8         | 0.001       | 0.81 | 0.899 | 0.593 |
| 16        | 0.01        | 0.699 | 0.712 | 0.565 |
| 32        | 0.01        | 0.716 | 0.784 | 0.455 |
| 64        | 0.1         | 0.738 | 0.697 | 0.701 |
| 128       | 0.01        | 0.877 | 0.678 | 0.782 |

- **Accuracy increased to ~90%** with weight initialization and batch normalization.
- Evaluated using **precision, recall, and mean average precision (mAP).**

## Work Plan

| Task | Duration | Responsible |
|------|---------|------------|
| Evaluate previous model | 3 days | Qadeer |
| Set up training framework | 2 days | Farhan |
| Research & implement model | 5 days | Samuel |
| Implement evaluation script | 3 days | Farhan |
| Tune hyperparameters | 1 day | Qadeer |
| Iterative testing | 5 days | All |

## Future Improvements

- Try **random search** for hyperparameter tuning.
- Explore **additional data augmentation techniques** beyond Gaussian noise.
- Optimize **model inference speed** for real-time deployment.

## References

1. [Introduction to Siamese Networks](https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942)
2. [MUCT Face Database](http://www.milbo.org/muct)
3. [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
4. [One-Shot Learning for Face Recognition](https://ieeexplore.ieee.org/document/8897021)

