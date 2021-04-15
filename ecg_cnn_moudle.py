import torch as th
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as tf


# 基本模型架构
class cnn_ecg_model(th.nn.Module):
    def __init__(self):
        super(cnn_ecg_model, self).__init__()
        # Args:
        #         in_channels (int): Number of channels in the input image
        #         out_channels (int): Number of channels produced by the convolution
        #         kernel_size (int or tuple): Size of the convolving kernel
        #         stride (int or tuple, optional): Stride of the convolution. Default: 1
        #         padding (int or tuple, optional): Zero-padding added to both sides of
        #             the input. Default: 0
        #         padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
        #             ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        #         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        #         groups (int, optional): Number of blocked connections from input
        #             channels to output channels. Default: 1
        #         bias (bool, optional): If ``True``, adds a learnable bias to the
        #             output. Default: ``True``
        # Examples:
        #
        #         >>> # With square kernels and equal stride
        #         >>> m = nn.Conv2d(16, 33, 3, stride=2)
        #         >>> # non-square kernels and unequal stride and with padding
        #         >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        #         >>> # non-square kernels and unequal stride and with padding and dilation
        #         >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        #         >>> input = torch.randn(20, 16, 50, 100)
        #         >>> output = m(input)

        #卷积核的表示方式不同 3和(3,1)不一样   3是一维 (3,1)是二维
        self.conv1 = th.nn.Conv1d(1, 4, kernel_size=21, stride=1, padding_mode='zeros')
        self.conv2 = th.nn.Conv1d(4, 16, kernel_size=23, stride=1, padding_mode='zeros')
        self.conv3 = th.nn.Conv1d(16, 32, kernel_size=25, stride=1, padding_mode='zeros')
        self.conv4 = th.nn.Conv1d(32, 64, kernel_size=12, stride=1, padding_mode='zeros')
        # pool
        self.pooling1 = th.nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling2 = th.nn.MaxPool1d(kernel_size=3, stride=2)
        self.pooling3 = th.nn.AvgPool1d(kernel_size=3, stride=2)
        # 全连接
        self.fc = th.nn.Linear(320, 5, bias=False)

    def forward(self, x):
        batch_size=x.shape[0]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pooling1(x)
        #print(x.shape)
        x = self.pooling2(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.pooling3(F.relu(self.conv3(x)))
        #print(x.shape)
        x=F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(batch_size, -1)
        print(x.shape)
        x = self.fc(x)

        return x
