import os
from model import *
from utils import *
import config as cfg
from dataset import *

import tensorflow as tf
from tensorflow.keras import Input, Model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def define_model(input_shape):
    inputs = Input(shape=input_shape)
    backbone = get_backbone_resnet50(input_shape=input_shape)
    global_features, global_outputs = GlobalNet(backbone=backbone)(inputs)
    refine_outputs = RefineNet()(global_features)
    model = Model(inputs=inputs, outputs=[global_outputs, refine_outputs])
    return model



input_shape = (256, 192, 3)

model = define_model(input_shape)
model.load_weights(cfg.save_path)

# Check its architecture
model.summary()

img_path = "coco/images/train2017_crop/000000000036_0.jpg"
img_orig = plt.imread(img_path)
img = tf.cast(tf.convert_to_tensor(img_orig-cfg.pixel_means), tf.float32) / 255.0
predicted = run_test(model, np.expand_dims(img, axis=0))
predicted = np.array(predicted).reshape(17, 2)
predicted[:, 0] = predicted[:, 0] * 4
predicted[:, 1] = predicted[:, 1] * 4
visualize_keypoints(np.stack((img_orig, img_orig), axis=0), keypoints=np.stack((predicted, predicted), axis=0))