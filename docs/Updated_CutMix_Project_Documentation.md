**CutMix:
Regularization Strategy to Train Strong Classifiers with Localizable Features** 

**Project Documentation**

**Paper:** Sangdoo Yun et al., *CutMix: Regularization
Strategy to Train*
*
Strong Classifiers with Localizable Features*, ICCV 2019.

**Paper link:** [https://arxiv.org/abs/1905.04899\\](https://arxiv.org/abs/1905.04899/)

**Implementation framework:** PyTorch

**Dataset:** CIFAR-10

**Project type:** Reduced-scale implementation and experimental study

**Status note:** This document includes the completed p = 0.5 ablation result. The project is a reduced-scale implementation, not an exact reproduction of the original paper's full experimental setup. The paper comparison below provides a reference result and explicitly notes the differences in scale and protocol.

---

**1. Abstract**

CutMix is a data-augmentation and regularization technique for image classification. It replaces a rectangular region of one training image with a patch from another image and mixes the training target in proportion to the image area contributed by each source.

In this project, the core CutMix operation is implemented in PyTorch and evaluated using a balanced subset of CIFAR-10. A Simple Convolutional Neural Network (SimpleCNN) is trained for 15 epochs using 3,000 training images and evaluated on 2,000 clean test images. An ablation study compares CutMix application probabilities p = 0.0, 0.5, and 1.0. In the recorded runs, p = 0.5 achieves 53.40% test accuracy and 1.3515 test loss, compared with 51.95% and 2.3736 for the baseline, and 50.95% and 1.4178 for full CutMix. These are preliminary observations from one run per configuration; they are not a full reproduction of the original paper's results.

**2. Introduction**

Image-classification models learn to map an input image to
one of a

fixed set of classes. A model may perform very well on its training

examples but generalize poorly to unseen examples. This behaviour is

known as **overfitting**.

Data augmentation attempts to improve generalization by
creating varied

training examples from existing data. Common approaches include random

cropping, flipping, Cutout, and Mixup. CutMix combines ideas from

patch-based augmentation and label mixing.

For CutMix, a rectangular patch from a second image is
pasted into a

first image. The target label is adjusted using the proportion of image

area retained from each source. This teaches the model to make

predictions from partially visible objects and may encourage it to use

evidence from multiple regions of an image.

**2.1 Problem statement**

The goal of this project is to implement the core CutMix
algorithm

independently in PyTorch and investigate its effect on

image-classification performance in a reduced-scale experiment.

**2.2 Objectives**

1. Implement      CutMix image mixing and area-based label weighting.
2. Train      a baseline CNN without CutMix.
3. Train      the same CNN architecture with CutMix.
4. Compare      clean-test-set accuracy and loss.
5. Perform      an ablation study by changing CutMix application
         probability.
6. Document      the reduced-scale setup and compare it carefully with a
         relevant result from the original paper.

**3. Background and intuition**

Consider two labeled images, image A and image B.

1. Start      with image A.
2. Select      a rectangular region.
3. Copy      the corresponding region from image B into image A.
4. Determine      how much of the final image comes from A and B.
5. Train      the classifier using a target that reflects both labels.

For example, if 70% of the final image remains from A and
30% is

replaced by B, the target assigns 0.7 weight to A's class and 0.3 weight

to B's class.

CutMix differs from related augmentation techniques:

---

Method Image operation Target-label treatment

---

Standard training Use the original image Original label

Cutout Mask/remove a region Usually keep the

original label

Mixup Blend two complete Mix labels using the

images blending coefficient

CutMix is an augmentation strategy; it is **not a new CNN**
**
architecture**. In this project, the CNN is the classifier and CutMix

changes the training examples supplied to it.

**4. Mathematical formulation**

Let two training examples be:

- (x_A):      first input image
- (y_A):      label of the first image
- (x_B):      second input image
- (y_B):      label of the second image

Let (M) be a binary mask. It indicates which pixels are
retained from

image A. The complementary region comes from image B.

**4.1 Mixed image**

[ \tilde{x}{=tex}=M\odot {=tex}x_A+(1-M)\odot {=tex}x_B ]

Here, (\odot{=tex}) denotes element-wise multiplication.
Where (M=1),

pixels come from A; where (M=0), pixels come from B.

**4.2 Mixed target**

[ \tilde{y}{=tex}=\lambda {=tex}y_A+(1-\lambda{=tex})y_B ]

Here, (\lambda{=tex}) is the fraction of the final image
retained from

A. For a rectangular patch, the initial value is sampled from a Beta

distribution and used to determine the patch size. Because the patch may

be clipped at image boundaries, the implementation recomputes the actual

value using the rectangle's final area:

[ \lambda {=tex}= 1-\frac{(x_2-x_1)(y_2-y_1)}{HW}{=tex} ]

where (H) and (W) are the image height and width, and

((x_1,y_1,x_2,y_2)) are the patch coordinates.

**4.3 Loss function**

The model predicts logits for the mixed image. The loss is
the weighted

sum of the two classification losses:

[

L=\lambda
{=tex}L(f(\tilde{x}{=tex}),y_A)+(1-\lambda{=tex})L(f(\tilde{x}{=tex}),y_B)

]

In PyTorch, this is implemented using two calls to the same

classification loss function, weighted by (\lambda{=tex}) and

(1-\lambda{=tex}).

**4.4 Beta distribution and parameters**

The mixing coefficient is sampled as:

[ \lambda{=tex}\_0

\sim {=tex}\operatorname{Beta}{=tex}(\alpha{=tex},\alpha{=tex})

]

The parameter (\alpha{=tex}) controls the distribution's
shape. The

project uses (\alpha=1.0{=tex}) for its current CutMix experiments.

The sampled coefficient controls the desired patch size; the final label

coefficient is then recalculated from the actual patch area.

Do not confuse these three quantities:

- **(p):**      probability that CutMix is applied to a training batch.
- **(\alpha{=tex}):**      parameter of the Beta distribution used to
        sample the initial mixing coefficient.
- **(\lambda{=tex}):**      actual proportion of the final image retained
        from image A.

**5. System architecture**

The project has a straightforward training pipeline:

CIFAR-10

   |

   v

Select balanced train/test subsets

   |

   v

Transform and normalize images

   |

   v

DataLoader creates mini-batches

   |

   v

Choose training mode

  
\|------------------------------|

   |                              |

   v                              v

Baseline: original batch     
CutMix: mix two images

   |                              |

   |                              v

   |                       Recalculate lambda

   |                              |

   |                       Weighted classification
loss

   |                              |

  
\|------------------------------|

                  |

                  v

             SimpleCNN

                  |

                  v

            
Predictions

                  |

                  v

       Backpropagation
and optimizer

                  |

                  v

          Trained
model

                  |

                  v

       Evaluation on
clean test images

                  |

                  v

       Test accuracy
and test loss

**5.1 Dataset pipeline**

The project uses CIFAR-10, which contains ten image classes.
The

reduced-scale subset is balanced:

- Training      subset: 3,000 images, 300 per class.
- Test      subset: 2,000 images, 200 per class.
- Image      shape: 3 colour channels, 32 × 32 pixels.

The selected training indices come from the official
training split, and

test indices come from the official test split. The splits are kept

separate. CutMix is applied only during training; test images remain

unmodified.

The subset indices should be saved so the same examples can
be selected

again. CIFAR-10 image files should not be committed to the Git

repository. The dataset is stored in Google Drive for persistence

between Colab runtime sessions.

**5.2 Preprocessing**

The current preprocessing converts images to tensors and
normalizes the

RGB channels using the CIFAR-10 channel statistics:

transforms.Compose([

   
transforms.ToTensor(),

   
transforms.Normalize(

        mean=(0.4914,
0.4822, 0.4465),

        std=(0.2470,
0.2435, 0.2616)

    )

])

The same preprocessing should be used for training and test
images.

CutMix pastes a patch between images that have undergone the same

transform.

**5.3 CNN architecture**

The project uses a small CNN called SimpleCNN. It consists
of two

convolutional blocks followed by a fully connected classifier.

---

Stage Operation Output shape for a

batch of 64

---

Input RGB image, 32 × 32 [64, 3, 32, 32]

Block 1 Conv2d: 3 → 32, 3 × 3, [64, 32, 16, 16]

padding 1; ReLU;

MaxPool2d(2)

Block 2 Conv2d: 32 → 64, 3 × 3, [64, 64, 8, 8]

padding 1; ReLU;

MaxPool2d(2)

Flatten Convert feature maps to [64, 4096]

a vector

Fully connected Linear: 4096 → 128; [64, 128]

ReLU

**Output Linear: 128 → 10 [64, 10]**

The output contains ten logits, one for each CIFAR-10 class.

CrossEntropyLoss consumes these logits and the class labels.

**5.4 Training configuration**

Current experimental settings:

Setting Value

---

Framework PyTorch

Dataset CIFAR-10 balanced subset

Training images 3,000

Test images 2,000

Batch size 64

Model SimpleCNN

Optimizer Adam

Learning rate 0.001

Epochs 15

CutMix alpha 1.0

Evaluation Clean test set, no CutMix

For a fair comparison, the baseline, full CutMix, and

probability-ablation runs should use the same data split, architecture,

epoch count, optimizer, and other training settings. Use and record

reproducible random seeds where possible. Because training is

stochastic, repeated runs with multiple seeds would provide a more

reliable comparison.

**6. CutMix algorithm**

**6.1 Algorithm in words**

**Inputs:** A batch of images, labels, and the parameter

(\alpha{=tex}).

1. Sample      an initial (\lambda{=tex}) from
         (\operatorname{Beta}{=tex}(\alpha{=tex},\alpha{=tex})).
2. Randomly      permute the batch to obtain a second image and label for
         each original example.
3. Compute      rectangle width and height using (\sqrt{1-\lambda}{=tex}).
4. Select      a random rectangle centre.
5. Clip      the rectangle coordinates to image boundaries.
6. Copy      the rectangle from the permuted images into the original
         images.
7. Recalculate      (\lambda{=tex}) from the actual pasted area.
8. Return      the mixed images, original labels, permuted labels, and
         corrected (\lambda{=tex}).
9. Run      the CNN on the mixed batch.
10. Compute      the weighted loss and update model parameters using
          backpropagation.

**6.2 Pseudocode**

function CUTMIX(images, labels, alpha):

    lambda_initial =
sample Beta(alpha, alpha)

    shuffled_indices =
random permutation of batch indices

    labels_a = labels

    labels_b =
labels[shuffled_indices]

    rectangle = random
rectangle whose area is based on lambda_initial

    clip rectangle to
image boundaries

    mixed_images =
copy(images)

    paste the
rectangle from shuffled images into mixed_images

    lambda_actual = 1 - actual_rectangle_area /
image_area

    return
mixed_images, labels_a, labels_b, lambda_actual

**6.3 Training with CutMix probability**

The ablation study uses an application probability (p):

for each training batch:

    draw a random
number r from Uniform(0, 1)

    if r < p:

        mixed_images,
labels_a, labels_b, lambda = CUTMIX(...)

        outputs =
model(mixed_images)

        loss = lambda
\* CE(outputs, labels_a)

             + (1 -
lambda) \* CE(outputs, labels_b)

    else:

        outputs =
model(original_images)

        loss =
CE(outputs, original_labels)

    clear previous
gradients

    backpropagate loss

    update model
parameters

The value of (p) is distinct from (\lambda{=tex}). When
(p=0), CutMix

is never applied; when (p=1), it is applied to every training batch. At

(p=0.5), each batch independently has a 50% chance of receiving CutMix,

so the exact fraction of augmented batches may differ slightly from 50%.

**7. Experimental methodology**

**7.1 Baseline experiment**

SimpleCNN is trained on original training images without CutMix. The model is evaluated on the untouched test subset to establish a baseline.

**7.2 CutMix probability ablation**

Three configurations were evaluated using the same stated dataset subset, SimpleCNN architecture, optimizer, learning rate, batch size, epoch count, and alpha value:

| Experiment | CutMix probability | Status |
|---|---:|---|
| Baseline | 0.0 | Completed |
| Partial CutMix | 0.5 | Completed |
| Full CutMix | 1.0 | Completed |

The main comparison metrics are test accuracy and test loss on clean images. Training accuracy under CutMix is not directly comparable to ordinary baseline training accuracy if it is calculated as hard-label correctness, because CutMix uses mixed images and area-weighted targets.

**8. Results**

**8.1 Test results**

| Configuration | CutMix probability | Test accuracy | Test loss |
|---|---:|---:|---:|
| Baseline CNN | 0.0 | 51.95% | 2.3736 |
| Partial CutMix | 0.5 | 53.40% | 1.3515 |
| Full CutMix | 1.0 | 50.95% | 1.4178 |

All three test evaluations use clean, unmodified test images.

**8.2 Ablation analysis**

Among the three recorded runs, p = 0.5 achieved the highest test accuracy (53.40%) and lowest test loss (1.3515). Its accuracy is 1.45 percentage points higher than the baseline and 2.45 percentage points higher than full CutMix.

These results suggest that, in this reduced-scale setup, applying CutMix probabilistically may offer a useful balance between original and augmented examples. This is a possible interpretation rather than a proven cause. Random initialization, data ordering, and sampled CutMix regions may affect the outcome. Since each configuration has only one run, the results do not establish statistical significance or a universally optimal probability.

The baseline's recorded training accuracy was 98.97% compared with 51.95% test accuracy, which is consistent with a large train–test gap. Full CutMix recorded 50.95% test accuracy and 1.4178 test loss: accuracy was 1.00 percentage point lower than baseline, while test loss was lower. Accuracy and loss measure different aspects of prediction, so these metrics can move differently.

**8.3 Interpretation cautions**

Do not claim that CutMix universally improves accuracy based on these three single runs. The experiment uses only a subset of CIFAR-10, a small custom CNN, and 15 epochs. For stronger evidence, repeat each configuration with multiple fixed seeds and report mean ± standard deviation.

**9. Comparison with the original paper**

This project is a reduced-scale implementation, not an exact reproduction of all experiments in the original paper. Yun et al. report a CIFAR-10 top-1 error of 2.88% for their CutMix configuration and 3.85% for their baseline in Table 7 [1]. Expressed as accuracy, these correspond to 97.12% and 96.15%, respectively.

| Item | Original paper result | This project |
|---|---|---|
| Dataset | CIFAR-10 | CIFAR-10 subset |
| Training/test data | Paper's experimental protocol | 3,000 training / 2,000 test images |
| Architecture | Architecture used in the paper's experiment | Custom SimpleCNN |
| Training schedule | Paper's reported training protocol | 15 epochs |
| CutMix result | 2.88% top-1 error (97.12% accuracy) | Best observed: 53.40% accuracy at p = 0.5 |
| Baseline result | 3.85% top-1 error (96.15% accuracy) | 51.95% accuracy at p = 0.0 |

**Important:** This table is a contextual comparison, not an apples-to-apples benchmark. The dataset scale, architecture, and training schedule differ substantially. The lower accuracy in this project should not be interpreted as a contradiction of the paper or as evidence that CutMix fails. A closer reproduction would require matching the paper's architecture, data, training schedule, augmentation configuration, and evaluation procedure.

**10. Limitations and failure analysis**

1. **Reduced      training data:** Only 3,000 training images are used, so
         the model sees far fewer examples than in the full CIFAR-10 training
         set.
2. **Small      architecture:** The custom CNN is intentionally simple and
         may not reproduce the performance of architectures used in the
         paper.
3. **Limited      training duration:** Fifteen epochs may not be sufficient
         for all configurations to converge.
4. **Aggressive      augmentation:** Applying CutMix to every batch can make
         the training task harder, especially with a small dataset and model.
5. **Stochasticity:**      Initialization, data order, and sampled
         rectangles can change the results. Multiple seeded runs would make
         conclusions more robust.
6. **Metric      interpretation:** Training accuracy with mixed targets is
         not directly comparable to ordinary training accuracy.
7. **Scale      mismatch:** Results from this reduced setup should not be
         presented as a reproduction of the paper's full-scale numerical
         results.

**11. Reproducibility and repository guidance**

Recommended repository organization:

CutMix-Implementation/

├── README.md

├── requirements.txt

├── src/

│   ├── cutmix.py

│   ├── model.py

│   ├── train.py

│   └── evaluate.py

├── notebooks/

│   └──
CutMix_Experiment.ipynb

├── experiments/

│   └── ablation/

│       ├── ablation.py

│       ├── ablation_results.csv

│       └──
ablation_notes.md

├── results/

│   ├── figures/

│   └── tables/

├── report/

│   └──
cutmix_report.pdf

└── presentation/

    └──
CutMix_Presentation.pptx

- Keep      dataset files out of Git; document how to download CIFAR-10.
- Save      selected subset indices and record the random seed.
- Record      Python and PyTorch versions and the device used.
- Save      experiment configurations and measured results.
- Ensure      the notebook can be run from top to bottom in a fresh Colab
        runtime.
- Save      the notebook to GitHub after changes; Colab does not
        automatically commit every edit to the repository.

**12. Suggested presentation structure**

For a presentation of approximately ten minutes:

1. **Title      and motivation** --- introduce CutMix and overfitting.
2. **Core      idea** --- show the rectangular patch and mixed target.
3. **Equations**      --- explain image mixing, label mixing, and weighted
         loss.
4. **Architecture**      --- describe CIFAR-10, preprocessing, and
         SimpleCNN.
5. **Algorithm      and implementation** --- explain the core CutMix
         function.
6. **Experimental      setup** --- state dataset subset, epochs, optimizer,
         and learning rate.
7. **Results**      --- compare baseline and full CutMix.
8. **Ablation**      --- compare p = 0.0, 0.5, and 1.0; p = 0.5 was best in the current single-run results.
9. **Comparison      with paper** --- clearly state differences in scale and
         protocol.
10. **Limitations      and conclusion** --- summarize what worked, what did
          not, and what remains uncertain.

**13. Likely viva questions**

**Q1. Is CutMix an architecture?**

No. It is a data-augmentation and regularization technique used while

training a model.

**Q2. Why are labels mixed?**

The final image contains pixels from both source images, so the target

reflects the fraction of area contributed by each source.

**Q3. Why recompute (\lambda{=tex})?**

The rectangle can be clipped at image boundaries. The actual pasted area

can therefore differ from the area implied by the initial sampled

coefficient.

**Q4. What is the difference between (p), (\alpha{=tex}),
and**
**
(\lambda{=tex})?**

(p) controls whether CutMix is applied; (\alpha{=tex}) shapes the Beta

distribution; (\lambda{=tex}) is the actual retained-area proportion

used in the loss.

**Q5. Why is test data not augmented with CutMix?**

The test set should measure performance on the intended clean, unseen

inputs rather than on artificially mixed images.

**Q6. Why can test loss decrease without test accuracy
increasing?**

Loss considers the confidence assigned to the correct class, while

accuracy only checks whether the top predicted class is correct.

**Q7. Did CutMix improve accuracy in this project?**

In the current single-run results, partial CutMix (p = 0.5) achieved 53.40% test accuracy, compared with 51.95% for the baseline and 50.95% for full CutMix. It performed best among the three tested settings in this run. Since each setting was evaluated once, repeated runs are needed before drawing robust conclusions.

**Q8. Why are the results different from the paper?**

The project uses a reduced dataset, a small custom CNN, and a short

training schedule. The experimental setup differs from the paper's

setup.

**14. References**

1. S.      Yun et al., "CutMix: Regularization Strategy to Train Strong
         Classifiers with Localizable Features," *Proceedings of the      IEEE/CVF*
         *International Conference on Computer Vision (ICCV)*, 2019.
         [https://arxiv.org/abs/1905.04899](https://arxiv.org/abs/1905.04899)
2. A.      Krizhevsky, "Learning Multiple Layers of Features from Tiny
         Images," CIFAR-10 dataset resource.
         [https://www.cs.toronto.edu/\\\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
3. PyTorch      documentation, [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)