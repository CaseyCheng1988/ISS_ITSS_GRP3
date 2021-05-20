import os, argparse
import numpy as np
import torch
from inference.predictor import SgmtPredictor
from inference.classifier import ViolenceAudNet
from utils.iou import nms
from utils.feature import ExtractSegAudFt
from utils.model import sigmoid
from utils.video import get_subclip

# construct argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="Movie filepath")
args = vars(ap.parse_args())

# Set up path and directory here
SYSTEM_DIR = os.path.dirname(os.path.realpath('__file__'))
print("Movie rating system directory: ", SYSTEM_DIR)

MODEL_DIR = os.path.join(SYSTEM_DIR, 'models')  ## to put your models here
print("Model directory: ", MODEL_DIR)

TEMP_DIR = os.path.join(SYSTEM_DIR, 'temp')

RPN_MODEL = os.path.join(MODEL_DIR, 'final_rpn.pth')

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


def __classify_violent_sgmts_visual__(sgmt_path):  ## Daniel to add violence classification
    pass


def __classify_violent_sgmts__(mov_path, sgmt_proposal, sgmt_score):
    mov_ext = mov_path.split('.')[-1]  # get movie extension

    # extract segment from full movie and save it in temporary folder
    aud_neg_score = []
    aud_pos_score = []
    aud_pred = []
    vis_neg_score = []
    vis_pos_score = []
    vis_pred = []
    for prop in sgmt_proposal:
        sgmt_path = os.path.join(TEMP_DIR, f'temp_clip.{mov_ext}')
        get_subclip(mov_path, sgmt_path, prop)

        # perform audio violence classification here
        a_neg, a_pos, a_pred = __classify_violent_sgmts_audio__(sgmt_path)
        aud_neg_score.append(a_neg)
        aud_pos_score.append(a_pos)
        aud_pred.append(a_pred)

        # perform visual violence classification here
        # v_neg, v_pos, v_pred = __classify_violent_sgmts_visual__(sgmt_path)         ## Daniel to modify the form of outputs here
        # vis_neg_score.append(v_neg)
        # vis_pos_score.append(v_pos)
        # vis_pred.append(v_pred)

    # remove temporary clip
    os.remove(sgmt_path)

    return aud_neg_score, aud_pos_score, aud_pred
    # return aud_neg_score, aud_pos_score, aud_pred, vis_neg_score,
    # vis_pos_score, vis_pred


def __nsfw_classification__(mov_path):  ## Daryl to add in nsfw classification and output here
    pass


def __rating_classification__(mov_path):
    # violence classification
    prop, score = __extract_sgmts__(mov_path, RPN_MODEL)
    ret = __classify_violent_sgmts__(mov_path, prop, score)
    aud_neg_score, aud_pos_score, aud_pred = ret[:3]
    # vis_neg_score, vis_pos_score, vis_pred = ret[3:]                              ## Daniel to modify output here

    # NSFW classification
    ret = __nsfw_classification__(mov_path)

    # SVM classification

################### main script running here #######################
input_movie = args["input"]
prop, score = __extract_sgmts__(input_movie, RPN_MODEL)
aud_neg_score, aud_pos_score, aud_pred = __classify_violent_sgmts__(
    input_movie, prop, score
)
print(aud_pred)