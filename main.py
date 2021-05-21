import os, argparse
import warnings

import numpy as np
import torch
from inference.predictor import SgmtPredictor
from inference.classifier import ViolenceAudNet, Videoto3D, PretrainedModel
from utils.iou import nms
from utils.feature import ExtractSegAudFt
from utils.model import sigmoid
from utils.video import get_subclip
warnings.filterwarnings('ignore')

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="Movie filepath")
args = vars(ap.parse_args())

# define clear screen function
def clear():
    if os.name == 'posix':
        os.system('clear')
    elif os.name == 'nt':
        os.system('cls')

# Set up path and directory here
SYSTEM_DIR = os.path.dirname(os.path.realpath('__file__'))
print("Movie rating system directory: ", SYSTEM_DIR)

MODEL_DIR = os.path.join(SYSTEM_DIR, 'models')  ## to put your models here
print("Model directory: ", MODEL_DIR)

TEMP_DIR = os.path.join(SYSTEM_DIR, 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

RPN_MODEL = os.path.join(MODEL_DIR, 'final_rpn.pth')
VIS_VIO_CLF_PATH = os.path.join(MODEL_DIR, 'model_c3d_v0815_daniel_depth40_400epoch.h5')

# instantiate audio violence classifier here
AUD_VIO_CLF_PATH = os.path.join(MODEL_DIR, 'final_audio_clf.pth')
AUD_VIO_CLF = ViolenceAudNet()
AUD_VIO_CLF.load_state_dict(torch.load(AUD_VIO_CLF_PATH))
AUD_VIO_CLF.eval()

# VIS_VIO_CLF = os.path.join(MODEL_DIR, '')                                       # Daniel to load your violence classification model

# extract segments of interests
# return final_proposal, final_score
# sgmt_proposal: list of numpy arrays of time intervals
# sgmt_score: list confidence scores
def __extract_sgmts__(mov_path, model_path):
    clear()
    print("\nStarting predictions of segments of interest......")
    predictor = SgmtPredictor(model_path)
    predictor.__setup_movie__(mov_path)
    confidence, location = predictor.__predict__()
    sgmt_proposal, sgmt_score = nms(confidence, location, thresh=0.2)
    print("Segments prediction done.")
    return sgmt_proposal, sgmt_score

def __classify_violent_sgmts_audio__(sgmt_path):
    # get mid-term audio features of segment
    sgmt_mt = ExtractSegAudFt(sgmt_path)

    # prepare mid-term format for torch model
    sgmt_mt = np.expand_dims(sgmt_mt, axis=0)
    sgmt_mt = torch.tensor(sgmt_mt).float()
    sgmt_vio_score = AUD_VIO_CLF.forward(sgmt_mt).detach().numpy()[0]
    neg_score, pos_score = sigmoid(sgmt_vio_score)
    pred = np.argmax(sgmt_vio_score)
    return neg_score, pos_score, pred

# Define parameter setting for visual C3D deep learning model
class Args:
    batch = 256
    epoch = 400
    nclass = 2 # 2 action categories
    depth = 40
    rows = 32
    cols = 32
    skip = True # Skip: randomly extract frames; otherwise, extract first few frames

param_setting = Args()
img_rows = param_setting.rows
img_cols = param_setting.cols
frames = param_setting.depth
channel = 1
vid3d = Videoto3D(img_rows, img_cols, frames)
nb_classes = param_setting.nclass

def __classify_violent_sgmts_visual__(clips, x_movie):  ## Daniel to add violence classification
    x_movie = x_movie.reshape((x_movie.shape[0], img_rows, img_cols, frames, channel))
    vis_neg_score, vis_pos_score, vis_pred = PretrainedModel(clips, x_movie, VIS_VIO_CLF_PATH)
    return vis_neg_score, vis_pos_score, vis_pred


def __classify_violent_sgmts__(mov_path, sgmt_proposal, sgmt_score):
    mov_ext = mov_path.split('.')[-1]  # get movie extension

    # extract segment from full movie and save it in temporary folder
    aud_neg_score = []
    aud_pos_score = []
    aud_pred = []
    X = []
    clips = []
    n_prop = len(sgmt_proposal)
    for i, prop in enumerate(sgmt_proposal):
        sgmt_path = os.path.join(TEMP_DIR, f'temp_clip.{mov_ext}')
        get_subclip(mov_path, sgmt_path, prop)

        # output to console
        clear()
        print(f"Performing violence classification: {i+1}/{n_prop} segments......")

        # perform audio violence classification here
        a_neg, a_pos, a_pred = __classify_violent_sgmts_audio__(sgmt_path)
        aud_neg_score.append(a_neg)
        aud_pos_score.append(a_pos)
        aud_pred.append(a_pred)

        # prepare frame sequences for visual violence classification
        varray = vid3d.get_data(sgmt_path, skip=True)
        X.append(varray)
        x_movie = np.array(X).transpose((0, 2, 3, 1))
        clips.append('clips%s' % i)

    # perform visual violence classification here
    vis_neg_score, vis_pos_score, vis_pred = __classify_violent_sgmts_visual__(clips, x_movie)
    # remove temporary clip
    os.remove(sgmt_path)
    clear()
    print("Done violence classification.")

    return aud_neg_score, aud_pos_score, aud_pred, vis_neg_score, vis_pos_score, vis_pred


def __nsfw_classification__(mov_path):  ## Daryl to add in nsfw classification and output here
    pass


def __rating_classification__(mov_path):
    # violence classification
    prop, score = __extract_sgmts__(mov_path, RPN_MODEL)
    ret = __classify_violent_sgmts__(mov_path, prop, score)
    aud_neg_score, aud_pos_score, aud_pred = ret[:3]
    vis_neg_score, vis_pos_score, vis_pred = ret[3:]                              ## Daniel to modify output here

    # NSFW classification
    ret = __nsfw_classification__(mov_path)

    # SVM classification

################### main script running here #######################
input_movie = args["input"]
prop, score = __extract_sgmts__(input_movie, RPN_MODEL)
aud_neg_score, aud_pos_score, aud_pred, vis_neg_score, vis_pos_score, vis_pred = __classify_violent_sgmts__(
    input_movie, prop, score
)
print(aud_neg_score)
print(aud_pos_score)
print(aud_pred)
print(vis_neg_score)
print(vis_pos_score)
print(vis_pred)