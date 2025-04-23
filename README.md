# Pneumonia Detection from Chest X-Rays Using CNNs

## Overview
Developed a deep learning model to classify chest X-ray images as pneumonia-positive or negative, enabling faster and more accurate diagnosis for radiologists in clinical settings.

## Dataset
Utilized the [RSNA Pneumonia Detection Challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) from Kaggle, containing 26,684 chest X-ray images labeled as pneumonia-positive or negative.

## Methodology
- **Preprocessing**: Resized images to 224x224 pixels and normalized pixel values to [0, 1]. Applied data augmentation (random rotation, flipping, and zooming) to improve model robustness.
- **Model**: Fine-tuned a pre-trained ResNet-50 model (pre-trained on ImageNet) with a custom classification head for binary classification.
- **Training**: Trained the model for 20 epochs using Adam optimizer (learning rate: 1e-4) on an AWS EC2 instance with GPU support (g4dn.xlarge).
- **Evaluation**: Used binary cross-entropy loss and evaluated performance with accuracy, AUC-ROC, and confusion matrix.
- **Visualization**: Generated Grad-CAM heatmaps to highlight regions of interest in X-ray images for interpretability.

## Results
- Achieved **85% accuracy** and **0.82 AUC-ROC** on the test set.
- Visualized model predictions with Grad-CAM heatmaps, showing focus on lung regions associated with pneumonia.
- Confusion matrix indicated low false-negative rate, critical for medical applications.

## Challenges
- **Class Imbalance**: Addressed imbalanced classes by using a weighted loss function and oversampling positive cases.
- **Overfitting**: Mitigated overfitting with data augmentation and dropout (0.5) in the fully connected layers.
- **Computational Constraints**: Optimized training by using mixed-precision training to reduce memory usage on AWS.

## Tools and Technologies
- **Languages**: Python
- **Libraries**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Albumentations
- **Platforms**: AWS EC2, Jupyter Notebooks
- **Version Control**: Git

## Code
[GitHub Repository](https://github.com/your-username/pneumonia-detection)

## Demo
[Streamlit App](https://your-streamlit-app-url.com)

## Impact
- Automated initial pneumonia screening, potentially reducing radiologist diagnosis time by 30%.
- Enhanced interpretability with Grad-CAM, increasing trust in model predictions for clinical use.
- Demonstrated applicability in resource-constrained healthcare settings by optimizing for efficient inference.
