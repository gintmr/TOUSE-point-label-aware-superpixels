##### spixel_utils #####
# This script contains utility functions including:
#
# -> find_mean_std: finds the mean and standard deviations for the Red, Green and Blue channel
# of an input image, such that the image can be normalized
#
# -> 

## IMPORTS ##
# Load necessary modules
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math

from skimage.color import rgb2lab
from skimage.util import img_as_float

from scipy import interpolate
import torch_scatter

### Functions ###
class img2lab(object):
    def __call__(self, img):
        # 将输入的图像转换为numpy数组
        img = np.array(img)
        # 将图像转换为浮点数格式，以便进行颜色空间转换
        flt_img = img_as_float(img)
        # 将RGB格式的图像转换为Lab颜色空间
        lab_img = rgb2lab(flt_img)
        # 返回转换后的Lab图像
        return lab_img
      
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#     def __call__(self, img):
#         assert isinstance(img, np.ndarray)
#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C x H x W
#         img = img.transpose((2, 0, 1))
#         return (torch.from_numpy(img))

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        print(f"Original image shape: {img.shape}")  # 添加调试输出
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if img.ndim == 3:  # 确保图像是三维的
            img = img.transpose((2, 0, 1))
        elif img.ndim == 4:  # 确保图像是四维的
            img = img.transpose((3, 0, 1, 2))
            img = np.squeeze(img, axis =  -1) # 删除最后一个维度
        else:
            raise ValueError(f"Expected image with 3 dimensions, got {img.ndim} dimensions.")
        return (torch.from_numpy(img))

class xylab(nn.Module):
    # 它定义了一个名为xylab的类，该类继承自nn.Module。这个类的目的似乎是将输入的Lab颜色空间的图像转换为XYLab颜色空间的特征。
    # Lab颜色空间是一种设备无关的颜色空间，而XYLab是Lab的一个变体，它将颜色信息（L和ab通道）与空间信息（X和Y通道）结合起来。
    def __init__(self, color_scale, pos_scale_x, pos_scale_y):
        super(xylab, self).__init__()
        self.color_scale = color_scale
        self.pos_scale_x = pos_scale_x
        self.pos_scale_y = pos_scale_y

    def forward(self, Lab):
        ########## compute the XYLab features of the batch of images in Lab ########
        # 1. rgb2Lab
        # 2. create meshgrid of X, Y and expand it along the mini-batch dimension
        #
        # Lab:   tensor (shape = [N, 3, H, W]): the input image is already opened in LAB format via the Dataloader defined #        in "cityscapes.py" 
        # lab格式为[N, 3, H, W]，代表N个图像，每个图像有3个通道，每个通道大小为HxW
        # XY:    tensor (shape = [N, 2, H, W])
        # XYLab: tensor (shape = [N, 5, H, W])
        
        N = Lab.shape[0]
        H = Lab.shape[2]
        W = Lab.shape[3]
        
        # Y, X = torch.meshgrid([torch.arange(0, H, out = torch.cuda.FloatTensor()), torch.arange(0, W, out = torch.cuda.FloatTensor())])
        # 使用torch.meshgrid函数创建X和Y坐标的网格，这些网格将被扩展到批次维度。
        Y, X = torch.meshgrid([torch.arange(0, H, out = torch.FloatTensor()), torch.arange(0, W, out = torch.FloatTensor())])
        # 此时Y和X的形状为[H, W]
        print("Yshape : ", Y.shape, "Xshape : ", X.shape)
        # print(Y, X)
        # print('X[None, None, :, :]', X[None, None, :, :].shape)
        # print('X[None, None, :, :].expand(N, -1, -1, -1)', X[None, None, :, :].expand(N, -1, -1, -1).shape)
        # 将X坐标扩展到批次维度，形成形状为[N, 1, H, W]的张量。
        X = self.pos_scale_x *  X[None, None, :, :].expand(N, -1, -1, -1)                            # shape = [N, 1, H, W]
        # print(X)
        # print(X.shape)
        # 将Y坐标扩展到批次维度，形成形状为[N, 1, H, W]的张量。
        Y = self.pos_scale_y *  Y[None, None, :, :].expand(N, -1, -1, -1)                            # shape = [N, 1, H, W]
        # 将Lab图像的颜色通道缩放，并转换为浮点类型，以确保所有输入张量在连接时具有相同的类型。
        Lab = self.color_scale * Lab.to(torch.float)                                               # requires casting as all input tensors to torch.cat must be of the same dtype

        # print(torch.cat((X, Y, Lab), dim = 1))
        # print(torch.cat((X, Y, Lab), dim = 1).shape)
        return torch.cat((X, Y, Lab), dim = 1), X, Y, Lab


def find_mean_std(img):
    # Finds the mean and standard deviation of each RGB channel of an input image

    total_pixel = img.shape[0] * img.shape[1]

    R_mean = np.sum(img[:,:,0]) / total_pixel
    G_mean = np.sum(img[:,:,1]) / total_pixel
    B_mean = np.sum(img[:,:,2]) / total_pixel

    R_std = math.sqrt( (np.sum((img[:, :, 0] - R_mean) ** 2)) / total_pixel)
    G_std = math.sqrt( (np.sum((img[:, :, 0] - G_mean) ** 2)) / total_pixel)
    B_std = math.sqrt( (np.sum((img[:, :, 0] - B_mean) ** 2)) / total_pixel)

    return [R_mean, G_mean, B_mean], [R_std, G_std, B_std]



def get_spixel_init(num_spixels, img_width, img_height):
    '''
    像素是一种将图像分割成多个区域（或“超像素”）的方法，每个区域包含相似的像素，并且这些区域通常用于图像处理和计算机视觉任务。
    '''


    # 初始化超像素
    
    # 计算超像素的数量和分布
    k = num_spixels
    # np.floor()函数将输入的浮点数向下取整，返回一个与输入形状相同的数组
    # np.sqrt()函数计算输入数组的平方根
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))  # 宽度方向的超像素数量 
    #. 要加上img_width与img_height的交错相除，可以调整长款不一的图片
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))  # 高度方向的超像素数量

    # 计算每个超像素的高度和宽度
    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)

    # 生成超像素中心点的坐标
    h_coords = np.arange(-spixel_height / 2., img_height + spixel_height - 1, spixel_height) 
    #.np.arange()函数返回一个等差数组，从-spixel_height / 2.到img_height + spixel_height - 1，步长为spixel_height
    #. 这个数组表示超像素中心点在高度方向上的位置，而img_height + spixel_height 需要 -1 是因为不需要最后一个点（最后一个点在图像外）
    w_coords = np.arange(-spixel_width / 2., img_width + spixel_width - 1, spixel_width)
    
    # 创建超像素索引值矩阵
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    #. np.arange()函数返回一个等差数组，从0到k_w * k_h - 1，步长为1
    #. np.reshape()函数将一维数组转换为二维数组，形状为(k_h, k_w)
    #. np.int32()函数将浮点数转换为32位整数

    spix_values = np.pad(spix_values, 1, 'symmetric')  # 对矩阵进行对称填充
    #. np.pad()函数在矩阵的边缘填充1个元素，填充方式为对称填充
    
    # 创建插值函数
    #. interpolate.RegularGridInterpolator()函数创建一个插值函数，用于在给定的坐标网格上进行插值
    #. 使用 h_coords 和 w_coords 作为坐标网格，spix_values 作为这些坐标点上的值。
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    # 生成图像所有像素的坐标
    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    #. np.meshgrid()函数创建一个坐标网格，用于在二维平面上进行插值
    #. indexing = 'ij'表示坐标网格的索引方式为矩阵索引
    #. meshgird后返回 的值为： [2, H, W],即两个矩阵，一个为整个图的i索引，一个是整个图的j索引
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()
    #. np.reshape()函数将二维坐标网格转换为一维数组
    #. .transpose()函数将一维数组转换为二维数组，形状为(img_width * img_height, 2)

    # 使用插值函数生成初始化的超像素图  
    spixel_initmap = f(all_points).reshape((img_height,img_width))
    ## 获取all_points个点的值，然后形状转换回去，就是按超像素划分的点了

    # 特征超像素初始化图与普通超像素初始化图相同
    feat_spixel_initmap = spixel_initmap
    
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]


def compute_init_spixel_feat(trans_feature, spixel_init, num_spixels):
    """
    给定输入的几个图像，根据超像素图，分别计算各自的特征


    初始化每个超像素的平均特征，使用CNN编码的特征 "trans_feature"

    参数:
    trans_feature: 形状为 [B, C, H, W] 的张量，B是批次大小，C是特征通道数，H和W是图像高度和宽度
    spixel_init: 形状为 [H, W] 的张量，包含初始化的超像素标签
    num_spixels: 超像素的数量

    返回:
    init_spixel_feat: 形状为 [B, K, C] 的张量，K是超像素数量
    """

    # 将trans_feature展平，保留批次和通道维度
    trans_feature = torch.flatten(trans_feature, start_dim=2)  # 形状变为 [B, C, N]，其中N = H * W

    # 交换维度，使N（像素数）成为第一维
    trans_feature = trans_feature.transpose(0, 2)  # 形状变为 [N, C, B]

    # 扩展spixel_init以匹配trans_feature的维度
    spixel_init = spixel_init[:, None, None].expand(trans_feature.size())  # 形状变为 [N, C, B]

    # 使用torch_scatter.scatter计算每个超像素的平均特征
    init_spixel_feat = torch_scatter.scatter(trans_feature, spixel_init, #。 reduce='mean'，如果多个特征向量具有相同的索引，它们的值将被平均，输出表示每个索引对应的聚合特征
                                             dim_size=num_spixels, reduce='mean', dim=0)  # 形状变为 [K, C, B]
    #.scatter()函数将trans_feature中的每个元素根据spixel_init的值分配到新的位置，然后计算每个超像素的平均值

    # 调整维度顺序以得到所需的输出形状
    result = init_spixel_feat.transpose(0, 2).transpose(1, 2)  # 形状变为 [B, K, C]
    
    return result