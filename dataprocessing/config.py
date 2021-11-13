import numpy as np

NAME="CPN"
data_shape = (256, 192)  # height, width
OUTPUT_SHAPE = (64, 48)  # height, width
NR_SKELETON = 17
IMG_WIDTH = 192
IMG_HEIGHT = 256
BATCH_SIZE = 24
EPOCHS = 150
NUM_KEYPOINTS = 17 * 2  # 17 pairs each having x and y coordinates
STEP_TO_SHOW_METRICS = 10
GRADIENT_CLIP_NORM = 5.0
MOMENTUM = 0.9
LEARNING_RATE = 0.01
LEARNING_OPTIMIZER = 'SGD'
LEARNING_MOMENTUM = 0.9

save_path = "checkpoint/keypoint_model_2"


pixel_means = np.array([[[122.7717, 115.9465, 102.9801]]]) # RGB

gk15 = (15, 15)
gk11 = (11, 11)
gk9 = (9, 9)
gk7 = (7, 7)


min_loss = 9999999