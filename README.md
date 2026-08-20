# Fashion-MNIST CNN — Trained, Evaluated, and Experimentally Deployed

**Step 3** in a series building from MLP fundamentals to CNNs.

Steps 1–2 focused on understanding neural-network mechanics using controlled synthetic data. This step moves to a **real image dataset**, introduces convolutional neural networks, and takes the model through a complete training and evaluation workflow.

The project also includes an **experimental Gradio deployment** to test the trained model outside the notebook and investigate what happens when a model trained on benchmark data is exposed to real-world images.

---

## Why This Step Matters

The earlier steps focused primarily on understanding the mechanics of neural networks:

* Forward propagation
* Backpropagation
* Non-linearity
* Classification
* Training dynamics

This step moves closer to an actual machine-learning engineering workflow:

> **Train → Validate → Evaluate → Save the best model → Test inference → Experiment with deployment**

The goal is not simply to achieve a high accuracy number.

It is to understand how a CNN behaves on a real dataset, how training decisions affect generalization, and what changes when the model leaves the controlled environment of the dataset.

---

## Dataset

**Fashion-MNIST** — 70,000 grayscale 28×28 images across 10 clothing categories:

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

The dataset contains:

* **60,000 training images**
* **10,000 test images**
* **10 classes**
* **28×28 grayscale images**

Dataset source:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

Fashion-MNIST was chosen because it is a well-established benchmark while remaining small enough for relatively fast experimentation and training.

---

## Architecture

```text
Input (1×28×28 grayscale)
        │
        ▼
Conv2d(1→32, 3×3)
        ↓
BatchNorm
        ↓
ReLU
        ↓
MaxPool(2×2)
        ↓
Dropout(0.25)
        │
        │ 32×14×14
        ▼
Conv2d(32→64, 3×3)
        ↓
BatchNorm
        ↓
ReLU
        ↓
MaxPool(2×2)
        ↓
Dropout(0.25)
        │
        │ 64×7×7
        ▼
Flatten
        ↓
Linear(3136→128)
        ↓
ReLU
        ↓
Dropout(0.5)
        ↓
Linear(128→10)
        │
        ▼
10-class logits
```

### Design Choices

| Component            | Choice                                         | Reason                                                                                                          |
| -------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Convolutional blocks | 2                                              | Enough depth to learn useful spatial features without making the network unnecessarily complex for 28×28 images |
| Batch Normalization  | After convolution                              | Stabilizes and improves training                                                                                |
| Dropout              | 0.25 in convolutional blocks, 0.5 in FC layers | Helps control overfitting                                                                                       |
| Augmentation         | Random crop with padding                       | Improves generalization without introducing unrealistic horizontal flips                                        |
| Loss                 | Cross-Entropy                                  | Standard objective for multi-class classification                                                               |
| Optimizer            | AdamW                                          | Provides adaptive optimization with decoupled weight decay                                                      |
| LR Scheduler         | OneCycleLR                                     | Controls the learning-rate schedule throughout training                                                         |
| Checkpointing        | Best validation accuracy                       | Ensures the deployed/exported model is not automatically the final-epoch model                                  |

---

## Training Pipeline

The training workflow follows:

```text
Fashion-MNIST
      │
      ▼
Data preprocessing
      │
      ▼
Training augmentation
      │
      ▼
CNN
      │
      ▼
Cross-Entropy Loss
      │
      ▼
AdamW + OneCycleLR
      │
      ▼
Validation
      │
      ▼
Best checkpoint
      │
      ▼
Test evaluation
      │
      ▼
Inference / Experimental deployment
```

The model is evaluated throughout training and the checkpoint with the **best validation accuracy** is retained.

This avoids accidentally exporting an overfit final-epoch model simply because it happened to be the last one trained.

---

## Results

| Metric              |         Value |
| ------------------- | ------------: |
| Validation Accuracy |       ~90–92% |
| Validation Loss     |        0.2459 |
| Dataset             | Fashion-MNIST |
| Number of Classes   |            10 |

The most challenging class boundaries are typically among visually similar clothing categories such as:

* Shirt
* T-shirt/top
* Pullover
* Coat

The notebook contains the detailed evaluation results and confusion matrix.

> Exact results can vary slightly between runs depending on training conditions and random initialization.

---

### Some Important Screenshots
![Loss and Accuracy for the Train and val](/src/Loss_Accuracy.png)
![Best validation accuracy](/src/best_validation.png)
![Local image description](/src/confusion_matrix.png)




## Experimental Deployment

The project includes an **experimental Gradio interface** for testing the trained model outside the notebook.

The purpose of this component is primarily to investigate the complete inference pipeline:

```text
Input Image
    ↓
Image Preprocessing
    ↓
Tensor Conversion
    ↓
CNN
    ↓
Class Prediction
    ↓
Confidence / Output
```

The deployment experiment helped expose an important practical ML concept:

### Benchmark performance ≠ real-world performance

Fashion-MNIST consists of small, centered, grayscale images with a very specific visual distribution.

A random photograph taken in the real world can differ significantly in:

* background
* lighting
* image resolution
* object scale
* orientation
* composition
* color
* camera characteristics

Therefore, strong performance on Fashion-MNIST does **not** guarantee equally strong predictions on arbitrary real-world photographs.

This experiment was useful precisely because it exposed the difference between **dataset performance and deployment conditions**.

---

## Repository Structure

```text
.
├── README.md
├── fashion_mnist_cnn.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── Loss_Accuracy.png
│   ├── best_validation.png
│   └── confussion_matrix.png
│
├── experimental_deploy/
│   ├── app.py
│   ├── debug_preprocessing.py
│   ├── model.py
│   ├── requirements.txt
│   └── fashion_cnn.pt
│
├── requirements.txt
└── LICENSE
```

### Directory Responsibilities

**`src/`**

Contains the primary training implementationa and assets:

* `data.py` — dataset loading, preprocessing, transforms, and DataLoaders
* `model.py` — CNN architecture
* `train.py` — training loop, validation, checkpointing, and evaluation
* `best_validation.png` -Screen shot of finding the best validation accuracy
* `Loss_Accuracy.png` - Loss and Accuracy curve according to the Train and validation
* `confusion_matrix` - Shows the how confusion between the classess

**`experimental_deploy/`**

Contains the experimental inference/deployment work:

* `app.py` — Gradio interface
* `debug_preprocessing.py` — preprocessing and inference debugging
* `model.py` — model definition used for experimental inference
* `fashion_cnn.pt` — exported trained weights

---

## Engineering Practices

### Shared Model Architecture

The model architecture used during training is kept consistent with the architecture used during inference.

This reduces the risk of a mismatch between:

> **the model that produced the weights**

and

> **the model that loads those weights during inference.**

### Best-Checkpoint Selection

The exported weights come from the best validation checkpoint rather than blindly using the final training epoch.

### Reproducibility

The project uses fixed seeds and pinned dependencies where practical to make experiments easier to reproduce.

### Separation of Concerns

Training, data processing, model definition, and experimental inference are kept separated rather than placing the entire workflow inside one notebook.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/MallikarjunJD/fashion-mnist-cnn.git
cd fashion-mnist-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py
```

The training script handles:

* Dataset loading
* Transformations
* Model creation
* Training
* Validation
* Learning-rate scheduling
* Checkpointing
* Evaluation

---

## What I Learned

This project went beyond simply implementing a CNN.

The main lessons were:

1. **CNNs are a natural progression from fully connected networks for image data.**
2. **Regularization and augmentation directly affect generalization.**
3. **Validation-based checkpointing matters when deciding which model to export.**
4. **A high benchmark accuracy does not automatically mean a model is ready for arbitrary real-world inputs.**
5. **Preprocessing is part of the model pipeline, not an afterthought.**
6. **Experimental deployment can reveal problems that are invisible inside a controlled notebook environment.**
7. **The gap between training a model and building something usable is an important part of ML engineering.**

---

## Series Roadmap

* [x] **Step 1** — MLP from scratch: regression on a synthetic linear function
* [x] **Step 2** — MLP on a classification task with a non-linear decision boundary
* [x] **Step 3** — CNN on real image data using Fashion-MNIST

---

## Author

**Mallikarjun Jadi**

Computer Science Engineering Student

Machine Learning Engineer | Full Stack Developer

## License

MIT
