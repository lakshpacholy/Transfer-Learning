## Practical No. 02 — Transfer Learning
### Fine-Tuning InceptionV3 for Plant Leaf Disease Classification

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green)
![Model](https://img.shields.io/badge/Model-InceptionV3-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Objective

Fine-tune a pre-trained **InceptionV3** model (trained on ImageNet) on the **PlantVillage** dataset to classify plant leaf diseases across 38 categories — by modifying the top layers and optimizing hyperparameters using a two-phase transfer learning strategy.

---

## Repository Structure

```
├── Lab_Assignment_2_Transfer_Learning (2).ipynb   # Main Colab notebook
├── README.md                                  # This file
```

---

## Dataset

| Property        | Details                                                                 |
|----------------|-------------------------------------------------------------------------|
| **Name**        | PlantVillage Dataset                                                   |
| **Source**      | [Kaggle — abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Classes**     | 38 (disease + healthy categories across multiple crops)                |
| **Total Images**| ~54,000 RGB leaf images                                                |
| **Task**        | Multi-class image classification                                       |

---

## Model Architecture

```
Input (224 × 224 × 3)
        ↓
InceptionV3 Base — pre-trained on ImageNet
(Frozen in Phase 1 / Partially unfrozen in Phase 2)
        ↓
GlobalAveragePooling2D
        ↓
Dense(512, activation='relu')
        ↓
BatchNormalization
        ↓
Dropout(0.4)
        ↓
Dense(38, activation='softmax')   ← Output: 38 disease classes
```

---

## Training Strategy — Two-Phase Transfer Learning

### Phase 1 — Feature Extraction (Base Frozen)
- All InceptionV3 base layers are **frozen**
- Only the new custom top layers are trained
- Learning Rate: `1e-3`
- Epochs: `10`

### Phase 2 — Fine-Tuning (Partial Unfreeze)
- Layers from index `249` onward are **unfrozen**
- Entire model retrained with a much lower learning rate to avoid catastrophic forgetting
- Learning Rate: `1e-5`
- Epochs: `10`

---

## Hyperparameters

| Parameter        | Value   |
|-----------------|---------|
| Image Size       | 224×224 |
| Batch Size       | 32      |
| LR — Phase 1     | 1e-3    |
| LR — Phase 2     | 1e-5    |
| Fine-Tune From   | Layer 249 |
| Dropout Rate     | 0.4     |
| Optimizer        | Adam    |
| Loss Function    | Categorical Crossentropy |

---

## Data Augmentation

Applied to training set only:

- Random rotation (±30°)
- Width & height shifts (±20%)
- Shear and zoom (±20%)
- Horizontal flip
- Nearest-fill for empty pixels

---

## How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload `Lab_Assignment_2_Transfer_Learning (2).ipynb` to Colab and set runtime to **GPU (T4)**.

### 2. Get Kaggle API Token
- Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
- Click **Create New Token** → downloads `kaggle.json`
- Upload it when prompted in Step 2 of the notebook

### 3. Run All Cells
Execute cells in order. The notebook will:
1. Download and extract the PlantVillage dataset
2. Preprocess and augment images
3. Build and train the InceptionV3 model (Phase 1 + Phase 2)
4. Plot training curves, confusion matrix, and sample predictions
5. Save the trained model as `.keras`

---

## Results

| Phase              | Train Accuracy | Val Accuracy |
|-------------------|---------------|-------------|
| Phase 1 (Frozen)   | 88.72%           | 91.42%        |
| Phase 2 (Fine-tuned)| 95.23%         | 96.77%         |


---


## References

- **Reference Paper:** [Frontiers — Fruit and vegetable leaf disease recognition based on a novel custom CNN](https://www.frontiersin.org/journals/plant-science)
- **Dataset Paper:** Hughes, D.P. & Salathé, M. (2015). An open access repository of images on plant health. [arXiv:1511.08060](https://arxiv.org/abs/1511.08060)
- **Model:** [InceptionV3 — Keras Applications](https://keras.io/api/applications/inceptionv3/)

---

## Author

- Lucky Sharma (202301040253)
- Laksh Pacholy (202301040263)
- Abhay Bhise (202402040016)

Subject: Deep Learning (PEC)
Lab Practical No.: 02
