# Lab 07 — Convolutional Neural Networks on Chest X-Rays

Lab 07 classifies chest X-ray images as NORMAL or PNEUMONIA using the
`Data/xray_dataset_covid19` folder, which holds a small train/test split organised into one
subdirectory per class.

## Why convolution instead of a Dense layer

A 224x224 greyscale image flattened into a Dense layer of 128 units needs over six million
weights in the first layer alone. A convolutional layer instead slides a small learned
kernel (typically 3x3) across the image and reuses the same weights at every position. This
gives two properties a Dense layer lacks:

- **Parameter sharing** — the same edge detector is useful everywhere in the image, so it
  is learned once rather than separately for each pixel position.
- **Translation invariance** — a feature is detected regardless of where in the frame it
  appears.

## Architecture

Three convolutional blocks, each `Conv2D(filters, 3x3, relu) -> MaxPooling2D(2x2)`, with
filter counts of 32, 64 and 128. Pooling halves the spatial dimensions at each block while
the filter count doubles, so the representation gets smaller spatially and deeper
semantically. The head is `GlobalAveragePooling2D -> Dense(64, relu) -> Dropout(0.5) ->
Dense(1, sigmoid)`.

## Data augmentation

The training set is tiny, so augmentation matters more than architecture here. The lab uses
random rotation up to 10 degrees, width and height shifts up to 10%, and zoom up to 10%.
Horizontal flips are **deliberately not used**: a chest X-ray has a definite left and right,
and the heart sits on one side, so flipping creates anatomically impossible training images.

## Transfer learning

The second half of the lab loads MobileNetV2 pretrained on ImageNet with `include_top=False`,
freezes the convolutional base, and trains only a new classification head. Despite ImageNet
containing no medical images, the low-level edge and texture filters transfer well, and the
transfer-learned model beats the from-scratch CNN by a wide margin on so small a dataset.
Fine-tuning then unfreezes the top block at a reduced learning rate of 1e-5.

## Evaluation

Accuracy is reported alongside a confusion matrix, because in a medical screening setting a
false negative — a missed pneumonia — is far more costly than a false positive.
