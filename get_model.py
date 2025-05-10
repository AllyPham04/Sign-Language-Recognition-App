import numpy as np
import pandas as pd
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split, GroupShuffleSplit 

import glob
import sys
import os
import math
import gc
import sklearn
import scipy
import json

from steps.constants import *
from steps.preprocess import *
from steps.set_up_model import *

train, N_SAMPLES = read_train_data()

train = preprocess_train(train)

sign2ord, ord2sign = take_encoded_sign(train)
preprocess_layer = PreprocessLayer()

if PREPROCESS_DATA:
    preprocess_data(train)

if USE_VAL:
    X_train, y_train, X_val, y_val, NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL,  validation_data = define_train()
else:
    X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data = define_train()

dummy_dataset = get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN)
X_batch, y_batch = next(dummy_dataset)

if not PREPROCESS_DATA and TRAIN_MODEL:
    y_pred = model.predict_on_batch(X_batch).flatten()
LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50) for step in range(N_EPOCHS)]
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)

if TRAIN_MODEL:
    # Clear all models in GPU
    tf.keras.backend.clear_session()

    # Get new fresh model
    model = get_model()

    # Actual Training
    history = model.fit(
            x=get_train_batch_all_signs(X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN),
            steps_per_epoch=len(X_train) // (NUM_CLASSES * BATCH_ALL_SIGNS_N),
            epochs=N_EPOCHS,
            # Only used for validation data since training data is a generator
            batch_size=BATCH_SIZE,
            validation_data=validation_data,
            callbacks=[
                lr_callback,
                WeightDecayCallback(model)
            ],
            verbose = VERBOSE,
        )
    
model.save_weights('model/model_sign_language.h5')

if USE_VAL:
    # Validation Predictions
    y_val_pred = model.predict({'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL}, verbose=2).argmax(axis=1)
    # Label
    labels = [ord2sign.get(i).replace(' ', '_') for i in range(NUM_CLASSES)]

# Landmark Weights
for w in model.get_layer('embedding').weights:
    if 'landmark_weights' in w.name:
        weights = scipy.special.softmax(w)

landmarks = ['pose_embedding', 'left_hand_embedding', 'right_hand_embedding']

# Define TFLite Model
tflite_keras_model = TFLiteModel(model, preprocess_layer)

demo_raw_data = load_relevant_data_subset(train['file_path'].values[0])
demo_output = tflite_keras_model(demo_raw_data)["outputs"]
demo_prediction = demo_output.numpy().argmax()

# Create Model Converter
keras_model_converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
# Convert Model
tflite_model = keras_model_converter.convert()
# Write Model
with open('model/model_sign_language.tflite', 'wb') as f:
    f.write(tflite_model)

# Predict
interpreter = tf.lite.Interpreter("model/model_sign_language.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

output = prediction_fn(inputs=demo_raw_data)
sign = output['outputs'].argmax()
