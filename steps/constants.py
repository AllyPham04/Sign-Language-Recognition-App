import numpy as np
import tensorflow as tf

# If True, processing data from scratch
# If False, loads preprocessed data
PREPROCESS_DATA = False
TRAIN_MODEL = True
# True: use 10% of participants as validation set
# False: use all data for training -> gives better LB result
USE_VAL = False

if PREPROCESS_DATA:
    ROOT_DIR = 'data/preprocessed'
else:
    ROOT_DIR = '.'

N_ROWS = 75
N_DIMS = 3
DIM_NAMES = ['x', 'y', 'z']
SEED = 42
NUM_CLASSES = 250
IS_INTERACTIVE = True
VERBOSE = 1 if IS_INTERACTIVE else 2

INPUT_SIZE = 64

BATCH_ALL_SIGNS_N = 2
BATCH_SIZE = 32
N_EPOCHS = 60
LR_MAX = 1e-3
N_WARMUP_EPOCHS = 0
WD_RATIO = 0.05
MASK_VAL = 4237

# LANDMARKS CONFIGURATION 
# Types of landmarks we're using
USE_TYPES = ['pose', 'left_hand', 'right_hand']

# Landmark indices in original data
POSE_IDXS0 = np.arange(33)  # 33 pose points
LEFT_POSE_IDXS0 = np.array([11, 13, 15, 17, 19, 21, 23])  # Left pose points
RIGHT_POSE_IDXS0 = np.array([12, 14, 16, 18, 20, 22, 24])  # Right pose points
LEFT_HAND_IDXS0 = np.arange(33, 54)  # 21 left hand points
RIGHT_HAND_IDXS0 = np.arange(54, 75)  # 21 right hand points

# Tổng hợp tất cả keypoints bạn dùng
CHOOSING_POSE_IDXS = np.concatenate((LEFT_POSE_IDXS0, RIGHT_POSE_IDXS0))
LANDMARK_IDXS0 = np.concatenate((CHOOSING_POSE_IDXS, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0))

N_COLS = LANDMARK_IDXS0.size

POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, CHOOSING_POSE_IDXS)).squeeze()
LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, LEFT_HAND_IDXS0)).squeeze()
RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS0, RIGHT_HAND_IDXS0)).squeeze()
HAND_IDXS = np.concatenate((LEFT_HAND_IDXS, RIGHT_HAND_IDXS))

POSE_START = 0
LEFT_HAND_START = 14
RIGHT_HAND_START = 35
ROWS_PER_FRAME = 75  # number of landmarks per frame

# LAYER CONFIGURATION
# Epsilon value for layer normalisation
LAYER_NORM_EPS = 1e-6

# Dense layer units for landmarks
HANDS_UNITS = 384
POSE_UNITS = 384

# final embedding and transformer embedding size
UNITS = 512

# Transformer
NUM_BLOCKS = 2
MLP_RATIO = 1

# Dropout
EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)

# Activations
GELU = tf.keras.activations.gelu

# Convolution for pose arm keypoints
USE_CONV_LAYER = True
CONV_FILTERS = 128
CONV_KERNEL_SIZE = 3
CONV_DROPOUT_RATIO = 0.2