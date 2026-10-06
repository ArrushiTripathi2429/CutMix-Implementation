# CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features

A PyTorch implementation and small-scale experimental reproduction of the paper:

**CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**  
Sangdoo Yun et al., ICCV 2019

Paper: https://arxiv.org/abs/1905.04899

---

##  Overview

This project implements and experimentally studies **CutMix**, a data augmentation and regularization technique proposed by Yun et al.

CutMix combines two training images by replacing a random rectangular region of one image with a region from another image. The labels are mixed according to the proportion of each image present in the final image.

The core CutMix equations are:

\[
\tilde{x} = M \odot x_A + (1-M) \odot x_B
\]

\[
\tilde{y} = \lambda y_A + (1-\lambda)y_B
\]

where:

- \(x_A, x_B\) are two training images.
- \(M\) is a binary mask representing the rectangular region.
- \(\lambda\) represents the proportion of image A remaining in the final image.
- \(y_A, y_B\) are the corresponding labels.

The loss is implemented as:

\[
L = \lambda L_A + (1-\lambda)L_B
\]

where \(L_A\) and \(L_B\) are the cross-entropy losses for the two original labels.

---

##  Project Objectives

The main objectives of this project are:

- Understand the CutMix methodology proposed in the original paper.
- Implement the core CutMix algorithm manually using PyTorch.
- Train a baseline CNN without CutMix.
- Train the same CNN using CutMix.
- Compare baseline and CutMix performance.
- Analyze the effect of CutMix on overfitting and generalization.
- Perform ablation experiments on CutMix parameters.
- Reproduce the core idea of the original paper at a smaller computational scale.

---

#  What is CutMix?

CutMix is a **training-time data augmentation and regularization technique**.

Given two images:

```text
Image A                  Image B
  Ship                     Truck

  ┌───────────────────┐
│                   │
│       SHIP        │
│      ┌───────┐    │
│      │ TRUCK │    │
│      │ PATCH │    │
│      └───────┘    │
│                   │
└───────────────────┘

CutMix

CutMix  copies an actual region from another image:

Image A + Real region from Image B

This preserves useful visual information while forcing the model to learn from multiple regions.

CIFAR-10
    ↓
Balanced 3K / 2K subset
    ↓
Data preprocessing
    ↓
DataLoader
    ↓
Simple CNN
    ↓
 ┌─────────────────────────┐
 │                         │
 ↓                         ↓
Baseline                CutMix
Training                Training
 │                         │
 ↓                         ↓
Evaluation              Evaluation
 │                         │
 └───────────┬─────────────┘
             ↓
       Result Comparison
             ↓
       Ablation Study

```

Dataset
CIFAR-10

The experiments use the CIFAR-10 dataset through torchvision.
CIFAR-10 contains:

10 classes
32 × 32 RGB images
50,000 training images
10,000 test images

CIFAR-10 contains:

10 classes
32 × 32 RGB images
50,000 training images
10,000 test images



