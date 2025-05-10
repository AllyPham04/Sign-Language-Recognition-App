import pandas as pd
import numpy as np
from constants import *
from steps.set_up_model import PreprocessLayer
from sklearn.model_selection import GroupShuffleSplit
import json

# Read Training Data
def read_train_data():
    if IS_INTERACTIVE or not PREPROCESS_DATA:
        train = pd.read_csv('data/train.csv').sample(int(5e3), random_state=SEED)
    else:
        train = pd.read_csv('data/train.csv')

    N_SAMPLES = len(train)

    return train, N_SAMPLES

def get_file_path(path):
    return f'./{path}'

def preprocess_train(train):
    N_SAMPLES = len(train)
    train['file_path'] = train['path'].apply(get_file_path)

    json_file = "data/sign_to_prediction_index_map.json"
    with open(json_file, "r") as f:
        mapping = json.load(f)

    # Map giá trị từ JSON vào cột 'sign'
    train["sign_ord"] = train["sign"].map(mapping)

    return train

def take_encoded_sign(train):
    # Dictionaries to translate sign <-> ordinal encoded sign
    sign2ord = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ord2sign = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    return sign2ord, ord2sign

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def get_data(file_path):
    data = load_relevant_data_subset(file_path)
    preprocess_layer = PreprocessLayer()
    # Process Data Using Tensorflow
    data = preprocess_layer(data)
    
    return data

# Get the full dataset
def extract_npy(train):
    # Create arrays to save data
    X = np.zeros([N_SAMPLES, INPUT_SIZE, N_COLS, N_DIMS], dtype=np.float32)
    y = np.zeros([N_SAMPLES], dtype=np.int32)
    NON_EMPTY_FRAME_IDXS = np.full([N_SAMPLES, INPUT_SIZE], -1, dtype=np.float32)

    # Fill X/y
    for row_idx, (file_path, sign_ord) in enumerate(train[['file_path', 'sign_ord']].values):
        # Log message every 5000 samples
        if row_idx % 5000 == 0:
            print(f'Generated {row_idx}/{N_SAMPLES}')

        data, non_empty_frame_idxs = get_data(file_path)
        X[row_idx] = data
        y[row_idx] = sign_ord
        NON_EMPTY_FRAME_IDXS[row_idx] = non_empty_frame_idxs
        # Sanity check, data should not contain NaN values
        if np.isnan(data).sum() > 0:
            print(row_idx)
            return data
        # break

    # Save X/y
    print(X.shape)
    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('NON_EMPTY_FRAME_IDXS.npy', NON_EMPTY_FRAME_IDXS)

    # X, y = shuffle(X, y, random_state=SEED)
    
    # Save Validation
    splitter = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=SEED)
    PARTICIPANT_IDS = train['participant_id'].values
    train_idxs, val_idxs = next(splitter.split(X, y, groups=PARTICIPANT_IDS))

    # Save Train
    X_train = X[train_idxs]
    NON_EMPTY_FRAME_IDXS_TRAIN = NON_EMPTY_FRAME_IDXS[train_idxs]
    y_train = y[train_idxs]
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('NON_EMPTY_FRAME_IDXS_TRAIN.npy', NON_EMPTY_FRAME_IDXS_TRAIN)

    # Save Validation
    X_val = X[val_idxs]
    NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS[val_idxs]
    y_val = y[val_idxs]
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('NON_EMPTY_FRAME_IDXS_VAL.npy', NON_EMPTY_FRAME_IDXS_VAL)

def define_train():
    if USE_VAL:
        # Load Train
        X_train = np.load(f'{ROOT_DIR}/X_train.npy')
        y_train = np.load(f'{ROOT_DIR}/y_train.npy')
        NON_EMPTY_FRAME_IDXS_TRAIN = np.load(f'{ROOT_DIR}/NON_EMPTY_FRAME_IDXS_TRAIN.npy')
        # Load Val
        X_val = np.load(f'{ROOT_DIR}/X_val.npy')
        y_val = np.load(f'{ROOT_DIR}/y_val.npy')
        NON_EMPTY_FRAME_IDXS_VAL = np.load(f'{ROOT_DIR}/NON_EMPTY_FRAME_IDXS_VAL.npy')
        # Define validation Data
        validation_data = ({ 'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, y_val)

        return X_train, y_train, X_val, y_val, NON_EMPTY_FRAME_IDXS_TRAIN, NON_EMPTY_FRAME_IDXS_VAL, validation_data
    else:
        X_train = np.load(f'{ROOT_DIR}/X.npy')
        y_train = np.load(f'{ROOT_DIR}/y.npy')
        NON_EMPTY_FRAME_IDXS_TRAIN = np.load(f'{ROOT_DIR}/NON_EMPTY_FRAME_IDXS.npy')
        validation_data = None

        return X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data
    
# Custom sampler to get a batch containing N times all signs
def get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, n=BATCH_ALL_SIGNS_N):
    # Arrays to store batch in
    X_batch = np.zeros([NUM_CLASSES*n, INPUT_SIZE, N_COLS, N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, NUM_CLASSES, step=1/n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([NUM_CLASSES*n, INPUT_SIZE], dtype=np.float32)
    
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == (i)).squeeze().astype(np.int32)
            
    while True:
        # Fill batch arrays
        for i in range(NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i*n:(i+1)*n] = X[idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n] = NON_EMPTY_FRAME_IDXS[idxs]
        
        yield {'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch}, y_batch
