# Sign Language Recognition ML-Ops

## Introduction

### Background
Approximately 90% of deaf children are born to hearing parents, many of whom may not be familiar with American Sign Language (ASL) ([kdhe.ks.gov](https://kdhe.ks.gov), [deafchildren.org](https://www.deafchildren.org)). Without early exposure to sign language, these children are at risk of developing **Language Deprivation Syndrome**, a condition caused by the lack of accessible language input during the critical language acquisition period. This can severely impact their ability to build relationships, succeed in education, and find future employment.

To support early learning of ASL, the **Isolated Sign Language Recognition Dataset (v1.0)** provides ~100,000 video samples of 250 signs performed by 21 Deaf signers. Hand, face, and body landmarks were extracted using **MediaPipe v0.9.0.1**, creating a rich set of features for training recognition models.

- `train_landmark_files/[participant_id]/[sequence_id].parquet`  
  Contains landmark data per video sequence. Extracted using MediaPipe Holistic.  
  > ⚠️ Not all frames necessarily contain detectable hand landmarks.

Each row includes:
- `frame`: Frame number in the original video  
- `row_id`: Unique identifier  
- `type`: One of `['face', 'left_hand', 'pose', 'right_hand']`  
- `landmark_index`: Index of the landmark  
- `x`, `y`, `z`: Normalized spatial coordinates (z may be unreliable)

- `train.csv`  
  - `path`: Path to the `.parquet` file  
  - `participant_id`: Unique signer ID  
  - `sequence_id`: Unique sequence ID  
  - `sign`: Label (sign name)

> ⚠️ Landmark data is not intended for identity recognition or biometric identification.

### Objectives

This project aims to build a model capable of **classifying isolated ASL signs** from landmark sequences. Using TensorFlow Lite and the MediaPipe Holistic pipeline, the trained model can infer signs in real time from user-uploaded or recorded videos.

The results may contribute to improving **PopSign\*** — an educational tool that helps families of deaf children learn and practice basic ASL for better communication.

### Technique

This system integrates **computer vision**, **machine learning**, and **interactive user interfaces** for isolated ASL recognition. Key components include:

- **MediaPipe Holistic**: Extracts 3D landmarks from face, body, and hands in each video frame  
- **OpenCV**: Handles video frame reading and landmark visualization  
- **TensorFlow Lite**: A lightweight model for real-time sign classification  
- **Gradio**: Provides a user-friendly web interface with:
  - A **tutorial video explorer** to browse example signs
  - A **practice tab** to upload or record videos, extract landmarks, and view predictions

The model returns the **top-10 predicted signs** with confidence scores and is optimized for smooth interaction across devices.

### Project Overview
![image](https://github.com/user-attachments/assets/6bc85e1b-e2b4-4724-9486-8198115009d9)

This project follows a structured deep learning pipeline for American Sign Language recognition. The main steps include:

1. **Input Processing**: Preprocessed keypoints from MediaPipe for pose, left hand, and right hand are used as input.
2. **Feature Embedding**: These keypoints are embedded using convolutional and dense layers.
3. **Attention Mechanism**: Embedded features are combined using attention layers.
4. **Sequence Modeling**: A Transformer encoder processes the sequence, utilizing:
   - Multi-Head Self Attention
   - Multi-Layer Perceptron (MLP)
5. **Pooling**: The processed sequence is pooled into a fixed-size representation.
6. **Classification**: The output is passed through a softmax layer to classify the sign.
7. **Training**: The model is trained using cross-entropy loss and the Adam optimizer.
8. **Deployment**: The trained TensorFlow Lite model is deployed on a Hugging Face Gradio interface for interactive use.

---

## 📂 Folder Structure
'''
.
├── data/
│   ├── videos/                        # Videos used for testing and reference
│   ├── ord2sign.csv                  # Maps sign index (sign_ord) to sign name
│   ├── sign_to_prediction_index_map/ # Index mapping for signs
│   └── train/                        # Training landmark files (as described above)
│
├── model/
│   └── model_sign_language.tflite    # Trained TensorFlow Lite model (~100K samples)
│
├── steps/                            # Source code for preprocessing and training
│   └── get_model.py                  # Script to define and train the model
│
├── app.py                            # Gradio interface app for deployment on Hugging Face
├── requirements.txt                  # Python dependencies for reproducing the environment
├── .gitignore                        # Git ignore file
└── README.md                         # Project documentation
'''
