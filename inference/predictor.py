import librosa
import numpy as np
import torch
from utils.audio import padAudio, paddedIdx
from utils.video import frameToTime, get_vid_fps
from utils.feature import extractSegmentMtAudFt
from utils.iou import findIouWithRefArr
from inference.classifier import RPN_1

# movie class
class Movie:
    def __init__(self, aud_path, sr, fps):
        self.aud, self.sr = librosa.load(aud_path, sr)
        self.fps = fps
        self.num_mt = 136
        self.start_t = 0
        self.end_t = (len(self.aud) - 1) / self.sr

    # build anchor points
    # dt: time between consecutive anchor points
    def _generate_anchor(self, dt):
        if self.sr * dt < 1:
            raise Exception("Class MoviePreparation:_generate_anchor --> dt cannot be smaller than 1/sr")
        ds = int(self.sr * dt)  # number of audio samples per dt
        numAnchors = int((len(self.aud) - 1) / ds)  # total anchor points
        self.anchors = np.arange(0, numAnchors + 1, dtype=np.int) * ds
        self.anchors_t = self.anchors/self.sr

    # build time scales around anchor
    def _generate_time_scales(self, start, end, step):
        self.t_scales = np.arange(start, end + step, step)

    # generate mid term features about anchors for each time scale
    def _generate_mt(self, mode='mirror', s_win_t=0.10, s_step_t=0.10):
        self.pad_mode = mode
        self.s_win_t = s_win_t
        self.s_step_t = s_step_t

        print("Generating mid-term features......")
        self.mt = np.zeros(shape=(self.num_mt, len(self.anchors), len(self.t_scales)))
        for s_idx, scale in enumerate(self.t_scales):
            print(f"At scale of {scale} s.....")
            width = int(self.sr * scale)  # total samples per scale

            # left and right total samples about anchor at current scale
            if (width % 2 == 0):
                l_off = width / 2
                r_off = width / 2
            else:
                l_off = (width - 1) / 2
                r_off = (width + 1) / 2

            # convert left and right sample numbers to integer
            l_off = int(l_off)
            r_off = int(r_off)

            # map original anchor index to the padded anchor index
            pad_anchors = paddedIdx(self.anchors, l_off)
            # pad audio signal using left and right offsets
            pad_aud = padAudio(self.aud, self.sr, l_off, r_off, mode=mode)

            # build mid-term features fir evert anchor at current scale
            for a_idx, pad_anchor in enumerate(pad_anchors):
                try:
                    self.mt[:, a_idx, s_idx] = extractSegmentMtAudFt(pad_aud[pad_anchor - l_off:pad_anchor + r_off + 1],
                                                                     self.sr, s_win_t, s_step_t)[0].reshape(-1)
                except:
                    print("Mid-term feature could not be computed for: ")
                    print("Anchor index: ", a_idx)
                    print("Scale: ", scale)

        print("Generation done.")

    # generate successful and fail segment candidates for training regional proposal module
    def _generate_candidates(self, gt_fr, high=0.7, low=0.1):
        self.success_threshold = high
        self.fail_threshold = low

        # map anchors sample index to time
        anchors_t = self.anchors / float(self.sr)

        # map ground truth frame intervals to time interval
        self.gt_t = frameToTime(gt_fr, self.fps)

        self.success = []
        self.fail = []
        for a_idx, anchor in enumerate(anchors_t):
            for s_idx, scale in enumerate(self.t_scales):
                # get left limit of anchor interval
                if anchor - scale / 2 <= self.start_t:
                    left = self.start_t
                else:
                    left = anchor - scale / 2

                # get right limit of anchor interval
                if anchor + scale / 2 >= self.end_t:
                    right = self.end_t
                else:
                    right = anchor + scale / 2

                # get ious of current anchor interval witl all groundtruth intervals
                ious = findIouWithRefArr(np.array([left, right]), self.gt_t)

                # include those with iou above success threshold in success list
                for i in np.where(ious >= high)[0]:
                    self.success.append((a_idx, s_idx, np.array([left, right]), self.gt_t[i], ious[i]))

                # include anchor interval in fail list if its ious with all groundtruth interval is below fail threshold
                if (ious <= low).all():
                    self.fail.append((a_idx, s_idx, np.array([left, right])))

        # build column names of success and fail list
        self.success_cols = ['anchor_index', 'scale_index', 'anchor_interval', 'groundtruth interval', 'iou']
        self.fail_cols = ['anchor_index', 'scale_index', 'anchor_interval']

    # generate all anchor intervals for inteference purporse
    def _generate_anchor_intervals(self):
        n_anchors = len(self.anchors_t)
        n_scales = len(self.t_scales)
        self.anchor_intervals = np.zeros((n_anchors, n_scales, 2))
        for a_idx, a_t in enumerate(self.anchors_t):
            for s_idx, s_t in enumerate(self.t_scales):
                left = a_t - s_t
                right = a_t + s_t
                if left <= self.start_t:
                    left = self.start_t
                if right >= self.end_t:
                    right = self.end_t
                self.anchor_intervals[a_idx, s_idx, :] = [left, right]

# segments predictor
class SgmtPredictor:
    def __init__(self, model_path):
        # set up RPN model
        self.model_path = model_path
        self.model = RPN_1()
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        # load mid-term parameters
        self.__set_mtparams__()

    # set mid-term parameters here
    def __set_mtparams__(self):
        MT_PARAMS = {'dt': 5.0,
                     'start': 2.0,
                     'end': 10.0,
                     'step': 4.0,
                     's_win_t': 0.50,
                     's_step_t': 0.50,
                     }
        self.dt = MT_PARAMS['dt']
        self.s_start = MT_PARAMS['start']
        self.s_end = MT_PARAMS['end']
        self.s_step = MT_PARAMS['step']
        self.s_win_t = MT_PARAMS['s_win_t']
        self.s_step_t = MT_PARAMS['s_step_t']

    # set up movie
    def __setup_movie__(self, mov_path, sr=22050):
        self.mov_path = mov_path
        self.mov_fps = get_vid_fps(mov_path)
        self.sr = sr

        # instantiate movie object
        self.mov_obj = Movie(self.mov_path,
                             sr=self.sr,
                             fps=self.mov_fps)

        # build movie anchor points
        self.mov_obj._generate_anchor(self.dt)

        # build time scales around each anchor
        self.mov_obj._generate_time_scales(self.s_start,
                                      self.s_end,
                                      self.s_step)

        # build movie mid-term features
        self.mov_obj._generate_mt(mode='mirror',
                             s_win_t=self.s_win_t,
                             s_step_t=self.s_step_t)

        # build all anchor intervals for inference
        self.mov_obj._generate_anchor_intervals()


    def __predict__(self):
        # get total anchors and scales
        n_anchors = len(self.mov_obj.anchors_t)
        n_scales = len(self.mov_obj.t_scales)

        # instantiate output arrays
        conf_score = np.zeros((n_anchors * n_scales, 2))
        cls_pred = np.zeros((n_anchors * n_scales,))
        rgr_pred = np.zeros((n_anchors * n_scales, 2))
        all_ints = np.zeros((n_anchors * n_scales, 2))

        i = -1
        for a_idx in range(n_anchors):
            for s_idx in range(n_scales):
                i += 1  # get current index

                # get current anchor interval and append to array
                a_int = self.mov_obj.anchor_intervals[a_idx, s_idx, :]
                all_ints[i, :] = a_int

                # get current mid term feature and convert to float32 for pytorch model
                mt = self.mov_obj.mt[:, a_idx, s_idx].astype(np.float32)
                xo, xr = self.model.forward(mt, obj=True)  # make predictions

                # get prediction on potentially violent classifier and
                # append prediction to list
                xo = xo.detach().numpy()
                conf_score[i, :] = xo
                xo = np.argmax(xo)
                cls_pred[i] = xo

                # map prediction by interval boundary regressor back to
                # time domain
                xr = xr.detach().numpy()
                xr = self.inverse_transform_time(xr, a_int)
                xr = np.array(xr)
                # clip prediction to min and max time of movie
                xr = np.clip(xr, self.mov_obj.start_t, self.mov_obj.end_t)
                rgr_pred[i, :] = xr

        pos_idx = np.where(cls_pred == 1)
        pos_conf_score = conf_score[pos_idx][:, 1]
        pos_rgr_pred = rgr_pred[pos_idx]
        return pos_conf_score, pos_rgr_pred

    # transform regressed xr back to time domain
    def inverse_transform_time(self, xr, t_a):
        x_a = (t_a[0] + t_a[1]) / 2
        w_a = t_a[1] - t_a[0]

        x = w_a * xr[0] + x_a
        w = w_a * np.exp(xr[1])

        return [x - w / 2, x + w / 2]