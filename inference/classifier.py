import numpy as np
import cv2
from torch import nn
from utils.model import activation_func, fc_function
import torch
import torch.nn.functional as F
from statistics import mean
import pandas as pd

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# architecture of RPN network
class RPN_1(nn.Module):
    def __init__(self, in_channels=136, out_channels=4, activation='relu', multiplier=2.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate = activation_func(activation)
        self.identity = nn.Identity()
        self.dropout = nn.Dropout(p=0.2)

        self.fc1, ret_channels = fc_function(self.in_channels, multiplier)
        self.fc2, ret_channels = fc_function(ret_channels, multiplier)
        self.fc3, ret_channels = fc_function(ret_channels, multiplier)
        self.fc4, ret_channels = fc_function(ret_channels, 1/multiplier)
        self.fc5, ret_channels = fc_function(ret_channels, 1/multiplier)
        self.fc6, ret_channels = fc_function(ret_channels, 1/multiplier)

        self.r_fc1, ret_r_channels = fc_function(ret_channels, 2)
        self.r_fc2, ret_r_channels = fc_function(ret_r_channels, 2)
        self.r_fc3 = nn.Linear(ret_r_channels, 2)

        self.o_fc1, ret_o_channels = fc_function(ret_channels, 2)
        self.o_fc2, ret_o_channels = fc_function(ret_o_channels, 2)
        self.o_fc3 = nn.Linear(ret_o_channels, 2)

    # obj (bool): whether violent or not
    def forward(self, x, obj):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        # U-structure of the network
        x_1 = self.fc1(x)
        x_1 = self.activate(x_1)

        x_2 = self.fc2(x_1)
        x_2 = self.activate(x_2)

        x_3 = self.fc3(x_2)
        x_3 = self.activate(x_3)

        x_4 = self.fc4(x_3)
        x_4 = self.activate(x_4)

        x_5 = self.fc5(x_4 + x_2)
        x_5 = self.activate(x_5)

        x_6 = self.fc6(x_5 + x_1)
        x_6 = self.activate(x_6)

        x_u = x_6 + x # output of U-structure

        # objectiveness network
        xo = self.o_fc1(x_u)
        xo = self.activate(xo)

        xo = self.o_fc2(xo)
        xo = self.activate(xo)

        xo = self.o_fc3(xo)

        # regression network
        if obj:
            xr = self.r_fc1(x_u)
            xr = self.activate(xr)

            xr = self.r_fc2(xr)
            xr = self.activate(xr)

            xr = self.r_fc3(xr)
            return xo, xr
        else:
            return xo

# Audio violence classifier
class ViolenceAudNet(nn.Module):
    def __init__(self):
        super(ViolenceAudNet, self).__init__()

        ################## convolution layers #####################

        # output channel is independent convolution of each input layer
        self.conv1d_1 = nn.Conv1d(in_channels=136,
                                  out_channels=136,
                                  kernel_size=3,
                                  padding=1,
                                  stride=1,
                                  dilation=1,
                                  groups=136,
                                  bias=True, )

        self.conv1d_2 = nn.Conv1d(in_channels=136,
                                  out_channels=136,
                                  kernel_size=3,
                                  padding=1,
                                  stride=1,
                                  dilation=1,
                                  groups=136,
                                  bias=True, )

        self.conv1d_3 = nn.Conv1d(in_channels=136,
                                  out_channels=272,
                                  kernel_size=3,
                                  padding=1,
                                  stride=1,
                                  groups=1,
                                  bias=True, )

        self.conv1d_4 = nn.Conv1d(in_channels=272,
                                  out_channels=544,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  dilation=1,
                                  groups=1,
                                  bias=True,
                                  )

        ################## pooling layers #######################
        self.pool1d = nn.MaxPool1d(kernel_size=2,
                                   stride=None,  # default as kernel_size
                                   padding=0,
                                   dilation=1,
                                   ceil_mode=True)

        ##################### fc layers ########################
        self.fc_1 = nn.Linear(in_features=1632,
                              out_features=408,
                              bias=True, )

        self.fc_2 = nn.Linear(in_features=408,
                              out_features=51,
                              bias=True, )

        self.fc_3 = nn.Linear(in_features=51,
                              out_features=2,
                              bias=True, )

    def forward(self, x):
        x = F.relu(self.conv1d_1(x))  # x reduced from 10 cols to
        x = F.relu(self.conv1d_2(x))
        x = F.relu(self.conv1d_3(x))
        x = self.pool1d(x)
        x = F.relu(self.conv1d_4(x))
        x = self.pool1d(x)
        x = nn.Flatten(1, -1)(x)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x


class Videoto3D:  # define vid3d class for the models

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def get_data(self, filename, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        bAppend = False
        if (nframe >= self.depth):
            if skip:
                frames = [x * nframe / self.depth for x in range(self.depth)]
            else:
                frames = [x for x in range(self.depth)]
        else:
            print("Insufficient %d frames in video %s, set bAppend as True" % (nframe, filename))
            bAppend = True
            frames = [x for x in range(int(nframe))]  # nframe is a float

        framearray = []

        for i in range(len(frames)):  # self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            # print(frame.shape)
            try:
                frame = cv2.resize(frame, (self.height, self.width))
            except:
                frame = prev_frame

            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            prev_frame = frame

        cap.release()

        if bAppend:
            while len(framearray) < self.depth:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            print("Append more frames in the framearray to have %d frames" % len(framearray))

        return np.array(framearray)


class Videoto3D:  # define vid3d class for the models

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def get_data(self, filename, skip=True):
        cap = cv2.VideoCapture(filename)
        nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        bAppend = False
        if (nframe >= self.depth):
            if skip:
                frames = [x * nframe / self.depth for x in range(self.depth)]
            else:
                frames = [x for x in range(self.depth)]
        else:
            print("Insufficient %d frames in video %s, set bAppend as True" % (nframe, filename))
            bAppend = True
            frames = [x for x in range(int(nframe))]  # nframe is a float

        framearray = []

        for i in range(len(frames)):  # self.depth):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
            ret, frame = cap.read()
            # print(frame.shape)
            try:
                frame = cv2.resize(frame, (self.height, self.width))
            except:
                frame = prev_frame

            framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            prev_frame = frame

        cap.release()

        if bAppend:
            while len(framearray) < self.depth:
                framearray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            print("Append more frames in the framearray to have %d frames" % len(framearray))

        return np.array(framearray)


# Define visual C3D deep learning model & processing of visual classification outputs
def DFbuild(clips, pred, non_violence_score, violence_score):
    clips.append('AVERAGE')

    pred.append(mean(pred))
    non_violence_score.append(mean(non_violence_score))
    violence_score.append(mean(violence_score))
    df = pd.DataFrame(list(zip(clips, pred, non_violence_score, violence_score)),
                      columns=['clips', 'prediction(1 means violent and 0 means not violent)', 'non_violence_score',
                               'violence_score'])
    df.to_csv('violence_scores.csv', index=False)


def scores(y_pred):
    non_violence_score = []
    violence_score = []
    for y in y_pred:
        # print(y)
        non_violence_score.append(y[0])
        violence_score.append(y[1])
    return np.argmax(y_pred, axis=1).tolist(), non_violence_score, violence_score

def PretrainedModel(clips,x_movie, model_path):
    c3d_model = Sequential()
    c3d_model.add(Conv3D(32, kernel_size=(3, 3, 3), input_shape=(x_movie.shape[1:]), padding='same'))
    c3d_model.add(Activation('relu'))
    c3d_model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    c3d_model.add(Activation('relu'))
    c3d_model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    c3d_model.add(Dropout(0.25))

    c3d_model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    c3d_model.add(Activation('relu'))
    c3d_model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    c3d_model.add(Activation('relu'))
    c3d_model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    c3d_model.add(Dropout(0.25))

    c3d_model.add(Flatten(name='flatten_feature'))
    c3d_model.add(Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    c3d_model.add(Dropout(0.2))
    c3d_model.add(Dense(2, activation='softmax'))

    c3d_model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.001, decay=1e-4 / 25),
                      metrics=['accuracy'])
    # c3d_model.summary()
    c3d_model.load_weights(model_path)
    y_pred = c3d_model.predict(x_movie, verbose=0)
    vis_pred, vis_neg_score, vis_pos_score = scores(y_pred)
    pred = vis_pred.copy()
    non_violence_score = vis_neg_score.copy()
    violence_score = vis_pos_score.copy()
    DFbuild(clips, pred, non_violence_score, violence_score)
    return vis_neg_score, vis_pos_score, vis_pred