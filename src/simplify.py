import time
import os
import numpy as np
import cv2 as cv
import torch
from lib.models.msra_resnet import get_pose_net

from lib.utils.image import get_affine_transform, transform_preds
from lib.utils.eval import get_preds, get_preds_3d


"""
all_pts=False,
alpha=0.99,
arch='msra_50',
batch_size=32,
data_dir='/home/ice/Documents/pytorch-pose-hg-3d/src/lib/../../data'
dataset='mpii'
debug=0
demo='../images'
ce=device(type='cpu')
disable_cudnn=False
eps=1e-06
epsilon=1e-08
exp_dir='/home/ice/Documents/pytorch-pose-hg-3d/src/lib/../../exp'
exp_id='default'
fit_short_side=False
flip=0.5
full_test=False
gpus=[-1]
heads={'hm': 16, 'depth': 16}
hide_data_time=False
hm_gauss=2
input_h=256
input_w=256
load_model='../models/fusion_3d_var.pth'
lr=0.001
lr_step=[90, 120]
metric='acc'
momentum=0.0
multi_person=False
num_epochs=140
num_output=16
num_output_depth=0
num_stacks=1
num_workers=4
output_h=64
output_w=64
print_iter=-1
ratio_3d=0
resume=False
root_dir='/home/ice/Documents/pytorch-pose-hg-3d/src/lib/../..'
rotate=30
save_all_models=False
save_dir='/home/ice/Documents/pytorch-pose-hg-3d/src/lib/../../exp/default'
scale=0.25
task='human2d'
test=False
val_intervals=5
weight_3d=0
weight_decay=0.0
weight_var=0
"""

JOINT_NUM = 16 # opt.num_output_depth, opt.num_output
MODEL_PATH = '../models/fusion_3d_var.pth' # opt.load_model
# MODEL_PATH = '../models/fusion_3d.pth' # opt.load_model
# MODEL_PATH = '../models/mpii.pth' # opt.load_model
LAYER_NUM = 50
HEADS = { 'hm': 16, 'depth': 16 }
LEARNING_RATE = 0.001
GPU_NUM = -1

CAMERA_WIDTH = 256
CAMERA_HEIGHT = 256
CAMERA_CENTER = np.array([CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0], dtype = np.float32)
CAMERA_SCALE = max(CAMERA_HEIGHT, CAMERA_WIDTH) * 1.0


MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

device_type = 'cpu' if GPU_NUM == -1 else 'cuda:{}'.format(GPU_NUM)
device = torch.device(device_type)

def create_model():
    model = get_pose_net(LAYER_NUM, HEADS)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    start_epoch = 1
    checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(MODEL_PATH, checkpoint['epoch']))

    state_dict = checkpoint['state_dict'] if type(checkpoint) == type({}) else checkpoint.state_dict()
    model.load_state_dict(state_dict, strict = False)

    model = model.to(device)
    model.eval()

    trans_input = get_affine_transform(CAMERA_CENTER, CAMERA_SCALE, 0, [CAMERA_WIDTH, CAMERA_HEIGHT])

    return model

class BodyPredictor:
    def __init__(self):
        self.model = create_model()

    def readImg(self, img):
        # inp = cv.warpAffine(img, trans_input, (CAMERA_WIDTH, CAMERA_HEIGHT), flags = cv.INTER_LINEAR)
        inp = img
        inp = (inp / 255. - MEAN) / STD
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        inp = torch.from_numpy(inp).to(device)
        out = self.model(inp)[-1]

        # pred = get_preds(out['hm'].detach().cpu().numpy())[0]
        # pred = transform_preds(pred, CAMERA_CENTER, CAMERA_SCALE, (CAMERA_WIDTH, CAMERA_HEIGHT))
        pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(),
                               out['depth'].detach().cpu().numpy())[0]

        return pred_3d


def test_single_image(predictor):
    img = cv.imread('../images/mpii_47.png')
    return predictor.readImg(img)

def test_multi_images(predictor):
    pathname = '../images'
    ls = os.listdir(pathname)
    for filename in sorted(ls):
        img = cv.imread(os.path.join(pathname, filename))
        predictor.readImg(img)

if __name__ == "__main__":
    body_predictor = BodyPredictor()

    start_time = time.time()
    # pred_3d = test_single_image(body_predictor, img)
    test_single_image(body_predictor)
    end_time = time.time() - start_time

    # print(pred_3d)
    print(end_time * 1000)

