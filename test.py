import os

from tensorflow.keras import Input, Model
from dataprocessing.dataset import *
from model.model import *
from model.utils import *
from glob import glob

import model.config as cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def define_model(input_shape):
    input = Input(shape=input_shape)
    backbone = get_backbone_resnet50(input_shape)
    global_fms, global_out = GlobalNet(backbone=backbone)(input)
    refine_out = RefineNet()(global_fms)
    model = Model(inputs=input, outputs=[global_out, refine_out])
    return model


def preprocess_images(paths):
    images = []
    for path in paths:
        img_orig = plt.imread(path)
        img_orig = cv2.resize(img_orig, (192, 256),
                              interpolation=cv2.INTER_CUBIC)
        img = (np.array(img_orig)-cfg.pixel_means) / 255.0
        images.append(img)

    return np.array(images)


input_shape = (256, 192, 3)

model = define_model(input_shape)
model.load_weights(cfg.save_path)

# Check its architecture
model.summary()

img_paths = glob("test_images/*")
images = preprocess_images(img_paths)
predicted = run_test(model, images)
predicted = np.array(predicted).reshape(-1, 17, 2)
predicted[:, :, 0] = predicted[:, :, 0] * 4
predicted[:, :, 1] = predicted[:, :, 1] * 4

visualize_keypoints(images, predicted)
