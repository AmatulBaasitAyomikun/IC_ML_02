📌 Facial Emotion Recognition using Deep Learning

📖 Project Overview

This project focuses on building a deep learning model to classify human facial expressions into multiple emotion categories. The goal was not only to train a model, but to understand and evaluate its learning behavior using training and validation metrics.

This project was completed as part of my InternCred internship.

🧠 Problem Statement

Facial emotion recognition is a challenging computer vision task due to variations in lighting, facial orientation, expression intensity, and class imbalance. This project aims to develop a baseline model that generalizes well while avoiding overfitting.

⚙️ Methodology

Image preprocessing and normalization

CNN-based architecture

Regularization using dropout

Data augmentation to improve generalization

Model evaluation using accuracy and loss curves

Deployment using Streamlit

📊 Model Performance

Test Accuracy: ~54%

Training and validation accuracy increase steadily

Training and validation loss decrease consistently

No evidence of overfitting

Signs of underfitting observed

The learning curves indicate that the model is stable and generalizes well, but requires further optimization to improve feature learning and overall accuracy.

📈 Key Insights

Learning curves are critical for diagnosing model behavior

Validation metrics can outperform training metrics when regularization is applied

Underfitting is preferable to overfitting at the baseline stage

Honest evaluation is more valuable than inflated performance claims

🚀 Deployment

The trained model has been deployed using Streamlit, allowing users to upload facial images and receive predicted emotion labels through a simple web interface.

🛠️ Tools & Technologies

Python

TensorFlow / Keras

OpenCV

NumPy, Matplotlib

Streamlit

🔮 Future Improvements

See the roadmap below.

🧩 Model Improvement Roadmap

1️⃣ Architecture Enhancements

Replace baseline CNN with transfer learning models:

ResNet50

MobileNetV2

EfficientNet

Fine-tune deeper layers instead of freezing all weights

2️⃣ Data-Level Improvements

Address class imbalance using:

Class weighting

Oversampling minority emotion classes

Increase dataset diversity (pose, lighting, age variation)

3️⃣ Training Optimization

Hyperparameter tuning:

Learning rate schedules

Batch size experimentation

Optimizer comparison (Adam vs AdamW)

Train for additional epochs with early stopping

4️⃣ Feature Learning Improvements

Higher-resolution inputs

Facial landmark extraction

Attention mechanisms to focus on key facial regions

5️⃣ Evaluation Improvements

Confusion matrix analysis

Per-class precision, recall, and F1-score

Emotion-wise error analysis

6️⃣ Deployment Enhancements

Real-time webcam inference

Confidence score for predictions

Model versioning for performance comparison
