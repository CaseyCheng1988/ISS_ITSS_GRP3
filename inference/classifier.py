from torch import nn
from utils.model import activation_func, fc_function
import torch
import torch.nn.functional as F

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