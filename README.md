# Sign Language Recognition ML-Ops

## 📌 Introduction

### 📊 Background
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

### 🎯 Objectives

This project aims to build a model capable of **classifying isolated ASL signs** from landmark sequences. Using TensorFlow Lite and the MediaPipe Holistic pipeline, the trained model can infer signs in real time from user-uploaded or recorded videos.

The results may contribute to improving **PopSign\*** — an educational tool that helps families of deaf children learn and practice basic ASL for better communication.

### 🧠 Technique

This system integrates **computer vision**, **machine learning**, and **interactive user interfaces** for isolated ASL recognition. Key components include:

- **MediaPipe Holistic**: Extracts 3D landmarks from face, body, and hands in each video frame  
- **OpenCV**: Handles video frame reading and landmark visualization  
- **TensorFlow Lite**: A lightweight model for real-time sign classification  
- **Gradio**: Provides a user-friendly web interface with:
  - A **tutorial video explorer** to browse example signs
  - A **practice tab** to upload or record videos, extract landmarks, and view predictions

The model returns the **top-10 predicted signs** with confidence scores and is optimized for smooth interaction across devices.

### 🚀 Project Overview
![image](https://github.com/user-attachments/assets/9bec01df-1702-4b80-b854-bd7bdda5a281)

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
```
.
├── data/
│   ├── videos/                        # Videos used for testing and inference
│   ├── ord2sign.csv                  # Maps sign index (sign_ord) to sign name
│   ├── sign_to_prediction_index_map/ # Index mapping for signs
│   └── train/                        # Training landmark files (as described above)
│
├── model/
│   └── model_sign_language.tflite    # Trained TensorFlow Lite model (~100K samples)
│
├── steps/                            # Source code for preprocessing and training
│   └── constants.py                  # Script to list constants in the model
│   └── preprocess.py                 # Script to define function for preprocessing the model
│   └── set_up_model.py               # Script to define the function for TFLite model
│
├── .gitignore                        # Git ignore file
├── app.py                            # Gradio interface app for deployment on Hugging Face
├── get_model.py                      # Script to define and train the model
├── requirements.txt                  # Python dependencies for reproducing the environment
└── README.md                         # Project documentation
```
## 🖥 Installation

To get started with the project, follow the steps below:

### Clone the Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/AllyPham04/ML-Ops
```

---

### Set up the Environment

Create a virtual environment and install the necessary packages:

```bash
conda create --name myenv
pip install -r requirements.txt
```

After creating all packages successfully, use that environment to run the system:

```bash
conda activate myenv
```

---

### Train Model

We have already trained and saved the model as `model_sign_language.tflite` in the `model/` folder.

If you want to start the project from the beginning, please download the preprocessed data in [**Sign Language Preprocessed Data**](https://www.kaggle.com/competitions/asl-signs), then run:

```bash
py get_model.py
```

This script will load the preprocessed data, train the model, and save the trained model to the `model/` directory.

> **Note:** If you want to directly test the system, you can SKIP this step.

### Hugging Face
We have deployed this system on HuggingFace. You can view it [here](https://allypham-signlanguagelearningapp.hf.space/).

## 📄 Results

![image](https://github.com/user-attachments/assets/15bd6896-a743-4ff8-9d6d-4d67500ec0f0)

After 60 epochs, the loss of the model has decreased significantly. From the loss of 4.4 in the first epoch, it has plummeted to 1.1 in the last epoch, proving its efficiency in training.

![image](https://github.com/user-attachments/assets/6d3b0331-e656-40f4-a7ef-4dc01ae458d4)

The model accuracy improved crucially after training. The starting accuracy lies at 0.18, but after 60 epochs, it reached an accuracy of over 0.9, close to 1, demonstrating its precision in predicting

![image](https://github.com/user-attachments/assets/f5f3aa17-b081-4ad7-9527-b5aa3db7e86d)

Looking at the weights of the model, we can see that the weights are allocated quite evenly between embedding points. The most important points are in the right hand, due to the dominance in the number of right handed people over left handed people. The dataset consists of keypoints extracted from videos of real people signing, so it is apparent that with the larger number of people using their right hand as their primary signing hand, the model is more sensitive towards right hand movements, hence the higher weights. Moreover, pose keypoints are not as important as hand keypoints, as ASL mainly focuses on hand movements. But overall, the weights of the embeddings do not differ much.

---

## 🔎 Future Works

For future development, we plan to:

- **Expand the model's vocabulary** by training on a larger dataset of ASL signs to improve recognition coverage.
- **Enhance the user interface** for a more intuitive and engaging user experience.
- **Support sentence-level recognition** by allowing users to string together multiple isolated signs into complete sentences, moving toward more natural and expressive sign language communication.

These improvements will bring the system closer to real-world applications and increase its usefulness for learners and families of deaf individuals.

## 🔏 References
1. (https://www.kaggle.com/competitions/asl-signs)
2. (https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485)
3. (https://www.kaggle.com/code/markwijkhuizen/gislr-tf-data-processing-transformer-training)
4. (https://www.youtube.com/watch?v=L-IaQch8KYY)
5. (https://www.youtube.com/watch?v=-UXYAAqm9fE)
