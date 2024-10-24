# SSN definition script
# This script just needed to build the feature extractor
# Pytorch implementation from: https://github.com/andrewsonga/ssn_pytorch
# Original paper is: https://varunjampani.github.io/ssn/

import torch
import torch.nn as nn
import torch.nn.functional as F
class crop(nn.Module):
    """
    将输入张量 x 在指定的维度 axis 上裁剪，使其大小与参考张量 ref 相同。这个模块可以在深度学习模型中用于调整张量的大小，以匹配另一个张量的空间维度。
    定义一个裁剪模块，用于裁剪输入张量x，使其在指定的维度axis上与参考张量ref的大小相同。
    默认情况下，axis为2，表示裁剪的是空间维度（H和W）。
    """

    def __init__(self, axis=2, offset=0):
        """
        初始化裁剪模块。
        :param axis: 指定裁剪的维度，默认为2，表示裁剪的是空间维度（H和W）。
        :param offset: 裁剪的偏移量，默认为0。
        """
        super(crop, self).__init__()
        self.axis = axis
        self.offset = offset

    def forward(self, x, ref):
        """
        前向传播，裁剪输入张量x，使其在指定的维度axis上与参考张量ref的大小相同。
        :param x: 输入张量。
        :param ref: 参考张量。
        :return: 裁剪后的输入张量。
        """
        for axis in range(self.axis, x.dim()):
            # 获取参考张量在当前维度的大小
            ref_size = ref.size(axis)
            # 生成裁剪的索引，起始位置为偏移量offset，结束位置为偏移量offset加上参考张量的大小
            indices = torch.arange(self.offset, self.offset + ref_size)
            # 将索引转换为与输入张量x相同的数据类型，并调整大小以匹配索引的大小
            indices = x.data.new().resize_(indices.size()).copy_(indices)
            # 使用索引对输入张量x进行裁剪
            x = x.index_select(axis, indices.long())
        return x

######################
#  Define the model  #
######################

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_pixel_features):
        super(CNN, self).__init__()
        
        ##############################################
        ########## 1st convolutional layer ###########
        self.conv1_bn_relu_layer = nn.Sequential()
        self.conv1_bn_relu_layer.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv1_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv1_bn_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ###### 2nd/4th/6th convolutional layers ######
        self.conv2_bn_relu_layer = nn.Sequential()
        self.conv2_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv2_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv2_bn_relu_layer.add_module("relu", nn.ReLU())

        self.conv4_bn_relu_layer = nn.Sequential()
        self.conv4_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv4_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv4_bn_relu_layer.add_module("relu", nn.ReLU())

        self.conv6_bn_relu_layer = nn.Sequential()
        self.conv6_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv6_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.conv6_bn_relu_layer.add_module("relu", nn.ReLU())
        
        ##############################################
        ######## 3rd/5th convolutional layers ########
        self.pool_conv3_bn_relu_layer = nn.Sequential()
        self.pool_conv3_bn_relu_layer.add_module("maxpool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.pool_conv3_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.pool_conv3_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels)) # the gamma and betas are trainable parameters of Batchnorm
        self.pool_conv3_bn_relu_layer.add_module("relu", nn.ReLU())

        self.pool_conv5_bn_relu_layer = nn.Sequential()
        self.pool_conv5_bn_relu_layer.add_module("maxpool", nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        self.pool_conv5_bn_relu_layer.add_module("conv", nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1))
        self.pool_conv5_bn_relu_layer.add_module("batchnorm", nn.BatchNorm2d(out_channels))
        self.pool_conv5_bn_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ####### 7th (Last) convolutional layer #######
        self.conv7_relu_layer = nn.Sequential()
        self.conv7_relu_layer.add_module("conv", nn.Conv2d(3 * out_channels + in_channels, num_pixel_features - in_channels, kernel_size = 3, stride = 1, padding = 1))
        self.conv7_relu_layer.add_module("relu", nn.ReLU())

        ##############################################
        ################### crop #####################
        self.crop = crop()

    def forward(self, x):

        conv1 = self.conv1_bn_relu_layer(x)
        conv2 = self.conv2_bn_relu_layer(conv1)
        conv3 = self.pool_conv3_bn_relu_layer(conv2)
        conv4 = self.conv4_bn_relu_layer(conv3)
        conv5 = self.pool_conv5_bn_relu_layer(conv4)
        conv6 = self.conv6_bn_relu_layer(conv5)
        # conv1 到 conv6：通过网络的各个层进行前向传播。

        # the input data is assumed to be of the form minibatch x channels x [Optinal depth] x [optional height] x width
        # hence, for spatial inputs, we expect a 4D Tensor
        # one can EITHER give a "scale_factor" or a the target output "size" to calculate thje output size (cannot give both, as it's ambiguous)
        conv4_upsample_crop = self.crop(F.interpolate(conv4, scale_factor = 2, mode = 'bilinear'), conv2)
        conv6_upsample_crop = self.crop(F.interpolate(conv6, scale_factor = 4, mode = 'bilinear'), conv2)
        # 对 conv4 和 conv6 进行上采样和裁剪，使其与 conv2 的空间尺寸相同。

        conv7_input = torch.cat((x, conv2, conv4_upsample_crop, conv6_upsample_crop), dim = 1)
        # 将输入数据 x、conv2、conv4_upsample_crop 和 conv6_upsample_crop 在通道维度上进行拼接
        conv7 = self.conv7_relu_layer(conv7_input)
        # 通过 conv7_relu_layer 进行前向传播，得到最终的输出 conv7。
        return conv7
