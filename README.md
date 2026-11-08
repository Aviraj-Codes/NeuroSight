# NeuroSight
### Intelligent Brain Imaging & Stroke Analysis

**An Explainable AI Framework for CT and MRI-Based Brain Image Classification**

NeuroSight is a deep learning-based brain image classification system designed to distinguish between Stroke and Control cases using Computed Tomography (CT) and Magnetic Resonance Imaging (MRI) images.

The project implements two independently trained ResNet18 convolutional neural networks, one for CT images and another for MRI images. It integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to provide visual explanations of model predictions through interpretable heatmaps.

The system is accessible through an interactive Streamlit web application that allows users to upload brain images, obtain classification predictions and visualize the regions influencing model decisions.

---

## Features

- **CT-Based Classification:** ResNet18 model trained to classify CT brain images into Stroke and Control categories.
- **MRI-Based Classification:** Independent ResNet18 model for MRI brain image classification.
- **Transfer Learning:** Utilizes pretrained ImageNet ResNet18 weights.
- **Explainable AI:** Integrates Grad-CAM to visualize image regions influencing predictions.
- **Interactive Web Application:** Streamlit-based interface for image uploads and inference.
- **Performance Evaluation:** Accuracy, Precision, Sensitivity, Specificity, F1-score and ROC-AUC.
- **Visual Analytics:** Confusion matrices, ROC curves and training-performance plots.

---

## System Architecture

NeuroSight follows two independent image classification pipelines.

```text
                 Input Brain Image
                        |
                Modality Selection
                   /         \
                  /           \
                CT             MRI
                |               |
          Image Preprocessing   Image Preprocessing
                |               |
          ResNet18 Model        ResNet18 Model
                |               |
          Binary Classifier     Binary Classifier
                |               |
          Stroke / Control      Stroke / Control
                |               |
          Grad-CAM              Grad-CAM
                |               |
          Visualization         Visualization
```

Each model independently processes its respective imaging modality. CT and MRI images are not artificially paired because a verified patient-level mapping between the modalities is unavailable.

---

## Dataset

**Dataset:** Multimodal Stroke Image Dataset

**Source:** Kaggle – `turkertuncer/multimodal-stroke-image-dataset`

The dataset contains 5,336 brain images from 230 reported subjects.

| Category | Number of Images |
|---|---:|
| Total Images | 5,336 |
| CT Images | 2,226 |
| MRI Images | 3,110 |
| Stroke Images | 2,695 |
| Control Images | 2,641 |

### Class Distribution

| Modality | Control | Stroke |
|---|---:|---:|
| CT | 1,297 | 929 |
| MRI | 1,344 | 1,766 |

### Dataset Partitioning

The original dataset partition is preserved.

| Split | Images |
|---|---:|
| Training | 4,009 |
| Testing | 1,327 |

The original training partition is further divided into training and validation subsets using an 80:20 split.

| Final Split | Images |
|---|---:|
| Training | 3,207 |
| Validation | 802 |
| Testing | 1,327 |

**Note:** The validation split is image-level. Although the original dataset paper reports patient-level partitioning, patient-level independence could not be independently verified from the available image filenames. Potential patient overlap within the derived training and validation subsets remains a limitation.

---

## Methodology

### 1. Image Preprocessing

All images undergo the following preprocessing operations:

- Resizing to 224 × 224 pixels.
- Conversion to RGB format.
- Tensor conversion.
- ImageNet normalization.
- Mild geometric augmentation during training.

### 2. Model Architecture

The project uses ResNet18, a convolutional neural network based on residual learning.

Two independent models are initialized using pretrained ImageNet weights.

The original classification layer is replaced with a single-neuron output layer for binary classification.

| Component | Configuration |
|---|---|
| Architecture | ResNet18 |
| Input Dimensions | 224 × 224 × 3 |
| Feature Extractor | Pretrained ResNet18 |
| Output Layer | Fully Connected Layer |
| Output Neurons | 1 |
| Output Interpretation | Binary Classification |
| Classification Classes | Stroke, Control |

### 3. Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.0001 |
| Loss Function | BCEWithLogitsLoss |
| Batch Size | 16 (GPU) |
| Maximum Epochs | 30 |
| Early Stopping Patience | 5 |
| Learning Rate Scheduler | ReduceLROnPlateau |
| Classification Threshold | 0.5 |

Class weighting is incorporated into the binary cross-entropy loss to address class imbalance.

### 4. Explainable AI Using Grad-CAM

Grad-CAM is integrated to improve the interpretability of model predictions.

It generates class activation heatmaps using gradients flowing into the final convolutional layer of ResNet18.

The heatmaps are superimposed on the original brain images to highlight regions influencing the predicted class.

---

## Evaluation Metrics

Both models are evaluated independently using:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Proportion of predicted Stroke cases that are actual Stroke cases |
| Sensitivity | Proportion of actual Stroke cases correctly identified |
| Specificity | Proportion of Control cases correctly identified |
| F1-score | Harmonic mean of Precision and Sensitivity |
| ROC-AUC | Ability to distinguish between Stroke and Control classes |

Evaluation results are exported to `performance_comparison.csv`.

Confusion matrices and ROC curves are generated to support comparative analysis.

---

## Application

NeuroSight provides an interactive web application developed using Streamlit.

### Application Workflow

1. Select the imaging modality: CT or MRI.
2. Upload a brain image.
3. The corresponding ResNet18 model processes the image.
4. View the predicted classification.
5. Examine the Stroke probability and prediction confidence.
6. Generate a Grad-CAM visualization.
7. View the experimental performance comparison.

---

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | Torchvision, PIL |
| Model Architecture | ResNet18 |
| Explainable AI | Grad-CAM |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-learn |
| Frontend | Streamlit |
| Development Environment | Google Colab, Jupyter Notebook |
| Version Control | Git, GitHub |

---

## Repository Structure

```text
NeuroSight/
│
├── app/
│   └── app.py
│
├── models/
│   ├── ct_model.pth
│   └── mri_model.pth
│
├── results/
│   ├── performance_comparison.csv
│   ├── ct_confusion.png
│   ├── mri_confusion.png
│   └── roc_comparison.png
│
├── notebooks/
│   └── Multimodal_Stroke_Detection.ipynb
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later.
- Git.
- pip package manager.

### 1. Clone the Repository

```bash
git clone https://github.com/Aviraj-Codes/NeuroSight.git
```

```bash
cd NeuroSight
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Model Checkpoints

Ensure the following files exist:

```text
models/ct_model.pth
models/mri_model.pth
```

### 5. Launch the Application

```bash
python -m streamlit run app/app.py
```

The application will be accessible at:

```text
http://localhost:8501
```

---

## Future Improvements

- Develop a multimodal architecture using verified patient-level CT–MRI image pairs.
- Investigate feature-level and decision-level fusion techniques.
- Explore advanced architectures such as EfficientNet, DenseNet and Vision Transformers.
- Improve generalization through external dataset validation.
- Investigate patient-level evaluation and leakage prevention.
- Extend explainability using additional XAI techniques.
- Optimize inference speed and model deployment.
- Integrate more comprehensive clinical image analysis capabilities.

---

## Limitations and Disclaimer

NeuroSight is an experimental academic research project.

The classification models operate on individual 2D brain images and have not been clinically validated. Their predictions should not be interpreted as medical diagnoses or used to make treatment decisions.

Grad-CAM visualizations indicate regions influencing model predictions; they do not establish the presence or location of clinically confirmed lesions.

The project also has limitations concerning patient-level independence and cross-modality image pairing.

Clinical deployment would require rigorous external validation, appropriate patient-level evaluation and assessment by qualified medical professionals.

---

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

## Author

**Aviraj Bandyopadhyay**

B.Tech – Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning  
Vellore Institute of Technology, Chennai

GitHub: [@Aviraj-Codes](https://github.com/Aviraj-Codes)

**Project:** NeuroSight – Intelligent Brain Imaging & Stroke Analysis

---

*Advancing Brain Image Analysis Through Deep Learning.*