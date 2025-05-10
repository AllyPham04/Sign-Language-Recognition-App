# Sign Language Recognition ML-Ops

## Introduction

### Background
Approximately 90% of deaf children are born to hearing parents, many of whom may not be familiar with American Sign Language (ASL) ([kdhe.ks.gov](https://kdhe.ks.gov), [deafchildren.org](https://www.deafchildren.org)). Without early exposure to sign language, these children are at risk of developing **Language Deprivation Syndrome**, a condition caused by the lack of accessible language input during the critical language acquisition period. This can severely impact their ability to build relationships, succeed in education, and find future employment.

To support early learning of ASL, the **Isolated Sign Language Recognition Dataset (v1.0)** provides ~100,000 video samples of 250 signs performed by 21 Deaf signers. Hand, face, and body landmarks were extracted using **MediaPipe v0.9.0.1**, creating a rich set of features for training recognition models.

## Dataset Structure

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
