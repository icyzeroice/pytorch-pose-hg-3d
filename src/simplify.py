import time
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

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_CENTER = np.array([CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0], dtype = np.float32)
CAMERA_SCALE = max(CAMERA_HEIGHT, CAMERA_WIDTH) * 1.0


MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)



def create_model():
    model = get_pose_net(LAYER_NUM, HEADS)
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)

    start_epoch = 1
    checkpoint = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(MODEL_PATH, checkpoint['epoch']))

    state_dict = checkpoint['state_dict'] if type(checkpoint) == type({}) else checkpoint.state_dict()
    model.load_state_dict(state_dict, strict = False)

    return model


def body_predictor(img):
    # """
    CAMERA_SCALE = max(img.shape[0], img.shape[1]) * 1.0
    CAMERA_CENTER = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    CAMERA_HEIGHT = img.shape[0]
    CAMERA_WIDTH = img.shape[1]
    # """

    device_type = 'cpu' if GPU_NUM == -1 else 'cuda:{}'.format(GPU_NUM)
    device = torch.device(device_type)
    model = create_model()
    model = model.to(device)
    model.eval()

    trans_input = get_affine_transform(CAMERA_CENTER, CAMERA_SCALE, 0, [CAMERA_WIDTH, CAMERA_HEIGHT])
    inp = img

    start_time = time.clock()
    # inp = cv.warpAffine(img, trans_input, (CAMERA_WIDTH, CAMERA_HEIGHT), flags = cv.INTER_LINEAR)
    print(inp.shape)
    inp = (inp / 255. - MEAN) / STD
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(device)
    out = model(inp)[-1]

    pred = None
    # pred = get_preds(out['hm'].detach().cpu().numpy())[0]
    # pred = transform_preds(pred, CAMERA_CENTER, CAMERA_SCALE, (CAMERA_WIDTH, CAMERA_HEIGHT))
    pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(),
                           out['depth'].detach().cpu().numpy())[0]
    end_time = time.clock() - start_time
    print(end_time * 1000)
    
    return pred, pred_3d


if __name__ == "__main__":
    img = cv.imread('../images/mpii_47.png')
    pred, pred_3d = body_predictor(img)

    print(pred)
    print(pred_3d)

