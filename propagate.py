##### Superpixel Generation #####
# This is main script that should be run, with all the specified parameters

# Load necessary modules
import torch
import torch.nn.functional as F
import json
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import numpy as np
from spixel_utils import *
from ssn import CNN
import os, argparse
from skimage.segmentation._slic import _enforce_label_connectivity_cython

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import matplotlib.colors as mcolors
from skimage.segmentation import mark_boundaries

from torchvision import transforms
import torchmetrics


def create_color_map(num_colors=256):
    color_map = np.zeros((num_colors, 3), dtype=np.uint8)
    
    for i in range(num_colors):
        r = (i * 7) % 256
        g = (i * 11) % 256
        b = (i * 13) % 256
        color_map[i] = [r, g, b]
    
    return color_map

# 创建颜色表
color_map_256 = create_color_map()

# 这个函数接受聚类结果，并输出每个像素对每个聚类的软成员度
def members_from_clusters(sigma_val_xy, sigma_val_cnn, XY_features, CNN_features, clusters):
    # 获取聚类结果的形状信息
    B, K, _ = clusters.shape
    # 创建sigma值的矩阵，用于后续计算
    sigma_array_xy = torch.full((B, K), sigma_val_xy, device=device)
    sigma_array_cnn = torch.full((B, K), sigma_val_cnn, device=device)
    
    # 提取聚类结果中的XY特征
    clusters_xy = clusters[:,:,0:2]
    # 计算XY特征与聚类结果XY特征之间的平方距离
    dist_sq_xy = torch.cdist(XY_features, clusters_xy)**2

    # 提取聚类结果中的CNN特征
    clusters_cnn = clusters[:,:,2:]
    # 计算CNN特征与聚类结果CNN特征之间的平方距离
    dist_sq_cnn = torch.cdist(CNN_features, clusters_cnn)**2

    # 根据距离计算软成员度，使用softmax函数
    soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * sigma_array_cnn**2)) , dim = 2)                # shape = [B, N, K] 
    # N是像素数量，K是聚类数量
    
    # 举出真实矩阵例子
    # 假设我们有一个批次大小为1的输入，包含3个聚类结果，每个聚类结果有2个XY特征和3个CNN特征
    # XY_features = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    # CNN_features = torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]])
    # clusters = torch.tensor([[[0.1, 0.2, 0.7, 0.8, 0.9], [0.3, 0.4, 1.0, 1.1, 1.2], [0.5, 0.6, 1.3, 1.4, 1.5]]])
    # sigma_val_xy = 0.5
    # sigma_val_cnn = 0.5
    # soft_memberships = members_from_clusters(sigma_val_xy, sigma_val_cnn, XY_features, CNN_features, clusters)
    # print(soft_memberships)
    
    return soft_memberships

# 函数用于对每个像素取最大类别似然，并在区域内强制连通性
# 该函数还根据"最小尺寸"计算将微小片段吸收到较大片段中
def enforce_connectivity(hard, H, W, K_max, connectivity = True):
    # 输入参数:
    # hard: 硬分配结果，形状为 [N]（是由上面的soft_memberships，计算torch.argmax(best_members, 2)得出的）
    # H: 图像高度
    # W: 图像宽度s
    # K_max: 最大超像素数量
    # connectivity: 是否强制连通性，默认为True

    # 设置批次大小为1
    B = 1

    # 将硬分配结果转换为numpy数组，并添加批次维度
    hard_assoc = torch.unsqueeze(hard, 0).detach().cpu().numpy()                                 # 形状为 [B, N]
    # 将一维数组重塑为三维数组，形状为 [B, H, W]
    hard_assoc_hw = hard_assoc.reshape((B, H, W))    

    # 计算每个分割的平均大小
    segment_size = (H * W) / (int(K_max) * 1.0)

    # 设置最小和最大分割尺寸
    min_size = int(0.06 * segment_size)  # 最小尺寸为平均分割大小的6%
    max_size = int(H*W*10)  # 最大尺寸为图像大小的10倍

    # 遍历每个批次（虽然这里批次大小为1）
    for b in range(hard_assoc.shape[0]):
        if connectivity:
            # 如果需要强制连通性，使用Cython实现的函数来强制标签连通性
            # _enforce_label_connectivity_cython的用法是：
            # 输入：
            # hard_assoc_hw[None, b, :, :]：形状为 [1, H, W] 的硬分配结果
            # min_size：最小分割尺寸
            # max_size：最大分割尺寸
            # 0：未使用
            # 输出：
            # spix_index_connect：形状为 [H, W] 的强制连通性结果
            
            spix_index_connect = _enforce_label_connectivity_cython(hard_assoc_hw[None, b, :, :], min_size, max_size, 0)[0]
        else:
            # 如果不需要强制连通性，直接使用原始的硬分配结果
            spix_index_connect = hard_assoc_hw[b,:,:]

    # 返回处理后的结果
    return spix_index_connect
    ## 最终结果的形状为[B, H, W]，含义为每个像素对应的连通性超像素索引。举例如下：
    ## 假设有3个超像素，每个超像素包含多个像素，那么最终结果的形状为[1, H, W]，其中每个像素的值为0, 1, 2等，表示该像素属于哪个超像素。

# 定义一个新的损失函数类，包含失真损失和冲突损失两个部分
class CustomLoss(nn.Module):
    def __init__(self, clusters_init, N, XY_features, CNN_features, features_cat, labels, sigma_val_xy = 0.5, sigma_val_cnn = 0.5, alpha = 1, num_pixels_used = 1000):
        super(CustomLoss, self).__init__()
        
        # 初始化类的属性
        self.alpha = alpha  # 失真损失的权重系数
        # nn.Parameter: 用于创建可训练的参数,clusters_init是初始化的聚类中心参数,requires_grad=True表示需要计算梯度
        self.clusters = nn.Parameter(clusters_init, requires_grad=True)  # 初始化可训练的聚类中心参数，维度为[B, K, C]
        B, K, _ = self.clusters.shape  # 获取聚类中心的形状信息
        
        self.N = N  # 总像素数
        
        # 设置XY特征和CNN特征的sigma值
        self.sigma_val_xy = sigma_val_xy
        self.sigma_val_cnn = sigma_val_cnn
        
        # 创建sigma值的张量，用于后续计算
        self.sigma_array_xy = torch.full((B, K), self.sigma_val_xy, device=device)
        # 形状为（B，K）,值为self.sigma_val_xy==0.5
        self.sigma_array_cnn = torch.full((B, K), self.sigma_val_cnn, device=device)
        
        # 存储特征和标签信息
        self.XY_features = XY_features  # XY坐标特征，每个像素的 XY 坐标特征，形状为 [B, N, 2]。这个特征用于计算与超像素中心的距离。

        self.CNN_features = CNN_features  #. CNN提取的特征，包含每个像素的 CNN 特征，形状为 [B, N, C]。这个特征用于计算与超像素中心的距离。
        #. 但是，通道数并不为5，而是有CNN相关网络参数计算得出的

        self.features_cat = features_cat  #。 组合后的特征，组合后的特征，通常是将 XY 特征和 CNN 特征拼接在一起，形状为 [B, N, C + 2]。这个特征用于计算失真损失。（也并不是简单的5）
        
        self.labels = labels  # 图像标签
        self.num_pixels_used = num_pixels_used  # 用于计算损失的像素数量

    def forward(self):
        # 这个函数计算超像素的失真损失和我们新提出的冲突损失
        
        # 输入:
        # 1) features: (torch.FloatTensor: shape = [B, N, C]) 定义每个图像的像素特征集
        
        # B 是批次维度
        # N 是像素数量
        # K 是超像素数量
        
        # 返回:
        # 1) 失真损失和冲突损失的加权和(我们在论文中使用lambda,但在编码中这意味着其他东西)
        
        # 随机选择一部分像素用于计算,以提高效率
        # randperm的用法: 返回一个从0到N-1的随机排列，随机排列的长度为num_pixels_used
        indexes = torch.randperm(self.N)[:self.num_pixels_used]

        ##################################### 失真损失 #################################################
        # 计算像素和超像素中心之间的距离,展开公式: (a-b)^2 = a^2-2ab+b^2 
        features_cat_select = self.features_cat[:,indexes,:]
        dist_sq_cat = torch.cdist(features_cat_select, self.clusters)**2

        # XY 分量
        clusters_xy = self.clusters[:,:,0:2] # 提取XY坐标特征（5个通道中的前两个）
        XY_features_select = self.XY_features[:,indexes,:] # XY_features_select 形状为 [B, num_pixels_used, 2]
        dist_sq_xy = torch.cdist(XY_features_select, clusters_xy)**2
        #。计算选定像素的 XY 特征与超像素中心之间的平方距离。

        # CNN 分量
        clusters_cnn = self.clusters[:,:,2:]
        CNN_features_select = self.CNN_features[:,indexes,:] # CNN_features_select 形状为 [B, num_pixels_used, C]
        dist_sq_cnn = torch.cdist(CNN_features_select, clusters_cnn)**2

        B, K, _ = self.clusters.shape
        
        # 计算软隶属度
        # 
        soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)

        # 距离由软隶属度加权
        dist_sq_weighted = soft_memberships * dist_sq_cat

        # 计算失真损失
        distortion_loss = torch.mean(dist_sq_weighted)

        ###################################### 冲突损失 ###################################################
        
        # 重塑标签张量
        labels_reshape = self.labels.permute(0,2,3,1).float()
        # 找到大于0的类别标签的索引(0表示未知类别)torch.gt(labels_reshape, 0) 返回一个布尔张量，表示 labels_reshape 中的每个元素是否大于 0。
        # 大于 0 的元素表示有效的类别标签（0 通常表示未知类别）。
        label_locations = torch.gt(labels_reshape, 0).float()#。float() 将布尔值转换为浮点数，结果将是一个形状为 [B, H, W, C] 的张量，其中有效标签的位置为 1.0，其他位置为 0.0。C应该是1才对，一个通道
        label_locations_flat = torch.flatten(label_locations, start_dim=1, end_dim=2) # 得出维度为[B, N, 1]

####################################
        # label_locations_flat = label_locations_flat[:, :, 0:1]
####################################
        XY_features_label = (self.XY_features * label_locations_flat)[0]  # shape = [N, 2]



        # try:
        #     # 进行逐元素相乘
        #     XY_features_label = (self.XY_features * label_locations_flat)[0]  # shape = [N, 2]
        #     print(XY_features_label)
        # except RuntimeError as e:
        #     print("RuntimeError:", e)  # 打印错误信息
        #     print("self.XY_features shape:", self.XY_features.shape)  # 再次打印 XY_features 的形状  [1, 1380060, 2]
        #     print(label_locations_flat[0][0], label_locations_flat[0][1], label_locations_flat[0][2])
        #     print("label_locations_flat shape:", label_locations_flat.shape)  #。再次打印 label_locations_flat 的形状  [1, 1380060, 3]======问题出现在此处
        #     raise  # 重新抛出异常以便后续处理



        ## abs() 函数返回每个元素的绝对值，sum(dim=1) 沿指定维度（这里是沿第1维）求和，> 0 返回一个布尔张量，表示每个元素是否大于0。
        non_zero_indexes = torch.abs(XY_features_label).sum(dim=1) > 0                          # shape = [N] 
        # 过滤掉未被选中的点
        XY_features_label_filtered = XY_features_label[non_zero_indexes].unsqueeze(0)           # shape = [1, N_labelled, 2]
        dist_sq_xy = torch.cdist(XY_features_label_filtered, clusters_xy)**2                    # shape = [1, N_labelled, K]

        CNN_features_label = (self.CNN_features * label_locations_flat)[0]                      # shape = [N, 15]
        CNN_features_label_filtered = CNN_features_label[non_zero_indexes].unsqueeze(0)         # shape = [1, N_labelled, 15]
        dist_sq_cnn = torch.cdist(CNN_features_label_filtered, clusters_cnn)**2                 # shape = [1, N_labelled, K]

        # 计算软隶属度
        soft_memberships = F.softmax( (- dist_sq_xy / (2.0 * self.sigma_array_xy**2)) + (- dist_sq_cnn / (2.0 * self.sigma_array_cnn**2)) , dim = 2)          # shape = [B, N_labelled, K]  
        soft_memberships_T = torch.transpose(soft_memberships, 1, 2)                            # shape = [1, K, N_labelled]

        ## [B, N, 1]，flatten，再选[0, :, :]， 就是[N, 1]
        labels_flatten = torch.flatten(labels_reshape, start_dim=1, end_dim=2)[0]               # shape = [N, 1]
        labels_filtered = labels_flatten[non_zero_indexes].unsqueeze(0)                         # shape = [1, N_labelled, 1] 

        # Use batched matrix multiplication to find the inner product between all of the pixels 
        # 矩阵乘法，soft_memberships 形状为 [1, N_labelled, K]，soft_memberships_T 形状为 [1, K, N_labelled]
        # torch.bmm 执行批量矩阵乘法，返回形状为 [1, N_labelled, N_labelled] 的张量
        innerproducts = torch.bmm(soft_memberships, soft_memberships_T)                         # shape = [1, N_labelled, N_labelled]
        ## 得出的innerproducts代表的是，每个像素与其他像素的相似度，越相似，值越大

        # Create an array of 0's and 1's based on whether the class of both the pixels are equal or not
        # If they are the the same class, then we want a 0 because we don't want to add to the loss
        # If the two pixels are not the same class, then we want a 1 because we want to penalise this


        check_conflicts_binary = (~torch.eq(labels_filtered, torch.transpose(labels_filtered, 1, 2))).float()      # shape = [1, N_labelled, N_labelled]
        #. 找出每个像素与其他像素的标签是否相同，相为同0，不同为1。
        #. 总结就是得到一个N-labelled*N-labelled的矩阵，所有被选中的，两两之间都为0，其他为1

        # try:
        #     # 进行逐元素比较
        #     # ~ 取反
        #     # torch.eq 逐元素比较
        #     # torch.transpose 交换维度  
        #     # 
        #     check_conflicts_binary = (~torch.eq(labels_filtered, torch.transpose(labels_filtered, 1, 2))).float()  # shape = [1, N_labelled, N_labelled]
        # except RuntimeError as e:
        #     print("RuntimeError:", e)  # 打印错误信息
        #     print("labels_filtered shape:", labels_filtered.shape)  # 再次打印 labels_filtered 的形状
        #     raise  # 重新抛出异常以便后续处理



        # Multiply these ones and zeros with the innerproduct array
        # Only innerproducts for pixels with conflicting labels will remain
        conflicting_innerproducts = torch.mul(innerproducts, check_conflicts_binary)           # shape = [1, N_labelled, N_labelled]

        # Find average of the remaining values for the innerproducts 
        # If we are using batches, then we add this value to our previous stored value for the points loss
        conflict_loss = torch.mean(conflicting_innerproducts)                                # shape = [1]

        return distortion_loss + self.alpha*conflict_loss, distortion_loss, self.alpha*conflict_loss

# 我们通过最小化我们的新颖损失函数来优化超像素中心位置
def optimize_spix(criterion, optimizer, scheduler, norm_val_x, norm_val_y, num_iterations=1000):
    
    best_clusters = criterion.clusters # 初始化超像素中心位置
    prev_loss = float("inf") # 初始化损失为无穷大

    for i in range(1,num_iterations):
        loss, distortion_loss, conflict_loss = criterion()
        # 每10步，我们将超像素中心的X和Y坐标限制在图像边界内
        if i % 10 == 0:
            with torch.no_grad():  # 不计算梯度，提高效率
                # 限制X坐标在[0, (image_width-1)*norm_val_x]范围内
                clusters_x_temp = torch.unsqueeze(torch.clamp(criterion.clusters[0,:,0], 0, ((image_width-1)*norm_val_x)), dim=1)
                # 限制Y坐标在[0, (image_height-1)*norm_val_y]范围内
                clusters_y_temp = torch.unsqueeze(torch.clamp(criterion.clusters[0,:,1], 0, ((image_height-1)*norm_val_y)), dim=1)
                # 将限制后的X、Y坐标与原始clusters的其他维度拼接
                clusters_temp = torch.unsqueeze(torch.cat((clusters_x_temp, clusters_y_temp, criterion.clusters[0,:,2:]), dim=1), dim=0)
            # 清空原始clusters数据
            criterion.clusters.data.fill_(0)
            # 用新的clusters_temp更新criterion.clusters
            criterion.clusters.data += clusters_temp 

        # 如果当前损失小于之前的最小损失，更新最佳clusters和最小损失
        if loss < prev_loss:
            best_clusters = criterion.clusters
            prev_loss = loss.item()

        # 反向传播计算梯度
        loss.backward(retain_graph=True)
        # 更新参数
        optimizer.step()
        # 清零梯度
        optimizer.zero_grad(set_to_none=True)
        # 更新学习率
        scheduler.step(loss)

        # 获取当前学习率
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']

        # 如果学习率小于0.001，提前结束优化
        if curr_lr < 0.001:
            break

    # 返回最佳的超像素中心位置
    return best_clusters


def show_mask(mask, ax, color):
    color = np.concatenate([color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def sparse_labels_to_mask(sparse_labels, H, W):
    sparse_labels = np.array(sparse_labels.cpu()) ##np.array() 将sparse_labels转换为NumPy数组
    mask = np.zeros((H, W)) 
    print("sparse_labels shape:", sparse_labels.shape)
    non_zero_indices = np.argwhere(sparse_labels.squeeze() > 10)  # 找出值不为0的点位
    print("non_zero_indices shape:", non_zero_indices.shape)
    ## non_zero_indices 的形状为 [num_points, 2],其中每一对点分别代表[y, x]

    return non_zero_indices



# 这个函数用于创建增强后的地面真实标签的RGB输出
def plot_propagated(NUM_CLASSES, save_path, propagated, original_image, sparse_labels, H, W, GT_img, connected, num_labels):
    ####### 函数用于将传播后的标签绘制为RGB图像 ########
    # 假设传播已由prop_to_unlabelled_spix_feat函数完成

    if NUM_CLASSES == 35:
        # UCSD Mosaics数据集的颜色映射
        # colors = [[167, 18, 159], [180, 27, 92], ..., [131, 69, 63]]  # 35个类别的颜色列表
        colors = [[123, 45, 67],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                [128, 128, 128],
                [255, 192, 203],
                [255, 165, 0],
                [75, 0, 130],
                [255, 20, 147],
                [0, 128, 128],
                [128, 0, 128],
                [255, 105, 180],
                [0, 0, 0],
                [255, 255, 255],
                [128, 0, 0],
                [0, 128, 0],
                [0, 0, 128],
                [128, 128, 0],
                [0, 128, 128],
                [192, 192, 192],
                [255, 140, 0],
                [255, 69, 0],
                [255, 228, 196],
                [255, 228, 181],
                [255, 222, 173],
                [255, 160, 122],
                [255, 99, 71],
                [255, 127, 80],
                [255, 228, 225],
                [240, 230, 140],
                [255, 228, 196],
                [255, 218, 185],
                [255, 228, 181],
                [255, 240, 245]
        ]

        bgr = np.array(colors) / 255.
        rgb = bgr[:,::-1]  # 将BGR转换为RGB

    elif NUM_CLASSES == 12:
        # CSIRO Segmentation数据集的颜色映射
        colors = [[0, 0, 0], [255, 0, 0], ..., [0, 0, 255]]  # 12个类别的颜色列表
        rgb = np.array(colors) / 255.
    elif NUM_CLASSES == 100:
        colors = {"NON_MIL": [0, 0, 0], "SSID": [30, 60, 90], "CAL_CCA_DC": [60, 120, 180], "NON_FREE": [90, 180, 14], "MALG": [120, 240, 104], "Dict": [150, 44, 194], "MASE_SMO_P": [180, 104, 28], "ACR-HIP": [210, 164, 118], "MASE_LRG_O": [240, 224, 208], "LSUB_SEDI": [14, 28, 42], "EAM_Sub": [44, 88, 132], "POR-MASS": [74, 148, 222], "OTH-SF": [104, 208, 56], "IRCI": [134, 12, 146], "SINV_SFC_O": [164, 72, 236], "PSEU": [194, 132, 70], "MACR_Cal_H": [224, 192, 160], "BRA_ARB_Ac": [254, 252, 250], "B_Monti": [28, 56, 84], "CVIR": [58, 116, 174], "ENSP": [88, 176, 8], "AGAR": [118, 236, 98], "SINV_SPO_E": [148, 40, 188], "DLAB": [178, 100, 22], "POR_NOD": [208, 160, 112], "SGRASS": [238, 220, 202], "Sand": [12, 24, 36], "SpMass": [42, 84, 126], "DSUB": [72, 144, 216], "BRA_FIN_Se": [102, 204, 50], "POR_Com_fi": [132, 8, 140], "POCI_CAU": [162, 68, 230], "BRA_TAB-Ac": [192, 128, 64], "Sediment": [222, 188, 154], "OCOM": [252, 248, 244], "FAV-MUS": [26, 52, 78], "MON_Cap_br": [56, 112, 168], "UTEN": [86, 172, 2], "EAM_RB": [116, 232, 92], "GORG": [146, 36, 182], "ACR-PE": [176, 96, 16], "ERHD": [206, 156, 106], "MASE_SML_O": [236, 216, 196], "POR-ENC": [10, 20, 30], "MLAG": [40, 80, 120], "Lvar": [70, 140, 210], "OTH-SINV": [100, 200, 44], "MACR_Cal_P": [130, 4, 134], "MASE_MEA_L": [160, 64, 224], "ALC-SF": [190, 124, 58], "POR-BRA": [220, 184, 148], "BRA_DIG_Ac": [250, 244, 238], "MPATU": [24, 48, 72], "ENGR1": [54, 108, 162], "Turfsa": [84, 168, 252], "TFP_RDG_Al": [114, 228, 86], "PPOR": [144, 32, 176], "Turf": [174, 92, 10], "SINV_SFC_A": [204, 152, 100], "POCI": [234, 212, 190], "SCplu": [8, 16, 24], "MACR_Fol_P": [38, 76, 114], "CCA": [68, 136, 204], "SINV_SPO_M": [98, 196, 38], "Unc": [128, 0, 128], "ZOAN": [158, 60, 218], "FISH": [188, 120, 52], "MASE_LRG_I": [218, 180, 142], "EAM_DHC": [248, 240, 232], "Mille": [22, 44, 66], "LSUB_SAND": [52, 104, 156], "OTH-HC": [82, 164, 246], "CYAN": [112, 224, 80], "ROSP": [142, 28, 170], "SINV_HYD": [172, 88, 4], "AMAT": [202, 148, 94], "BRA_SMO_Po": [232, 208, 184], "MINV_Dia": [6, 12, 18], "ACR-TCD": [36, 72, 108], "BRA_VER_Po": [66, 132, 198], "MADR": [96, 192, 32], "PASTR": [126, 252, 122], "ACR-OTH": [156, 56, 212], "ACR-BRA": [186, 116, 46], "MON_Cap_pl": [216, 176, 136], "ERGR": [246, 236, 226], "MASE_MEA_O": [20, 40, 60], "BRA_RND_St": [50, 100, 150]}
    elif NUM_CLASSES == 256:
        colors = color_map_256
        rgb = np.array(colors) / 255.
    else:
        print("我们没有存储该类别数量的颜色映射，请指定 - 请查看plot_propagated函数。")



# ############################################################################################################
#     mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', rgb)

#     mymap.set_bad(alpha=0)  # 设置颜色映射如何处理"坏"值
#     plt.register_cmap(name='my_colormap', cmap=mymap)

#     plt.set_cmap('my_colormap')
    
#     # 设置颜色映射的归一化范围
#     norm = mcolors.Normalize(vmin=0, vmax=NUM_CLASSES-1)
#     m = cm.ScalarMappable(norm=norm, cmap=mymap)
  
#     # 将传播后的标签转换为颜色
#     ## color_mask 的形状为 [H, W, 4]
#     color = m.to_rgba(propagated)

#     # 将原图转换为RGBA格式
#     original_image_rgba = np.array(original_image.convert("RGB"))
#     GT_img_L = np.array(GT_img.convert('RGB'))

#     # # 创建一个透明掩码
#     # alpha_mask = color_mask[..., 3]  # 获取颜色掩码的alpha通道
#     # color_mask[..., 3] = alpha_mask * 0.5  # 设置掩码的透明度

#     # # 合成原图和颜色掩码
#     # # color_mask[..., 3:] 的形状为 [H, W, 1]
#     # # color_mask[..., :3] 的形状为 [H, W, 3]
#     # # 1 - color_mask[..., 3:] 的形状为 [H, W, 1]
#     # # color_mask[..., :3] * color_mask[..., 3:] 的形状为 [H, W, 3]
#     # combined_image = (0.8) * original_image_rgba + color_mask * 0.95

#     # # 创建并保存图像
#     # fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     # plt.axis('off')
#     # plt.imshow(combined_image.astype(np.uint8))
#     # plt.savefig(save_path+".jpg", bbox_inches='tight')
#     # plt.close()
#     non_zero_indices = sparse_labels_to_mask(sparse_labels, H, W)

#     fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     plt.xlim(0, W)
#     plt.ylim(H, 0)
#     # figsize 是图像的尺寸，facecolor 是图像的背景颜色，frameon 是是否显示边框
    
#     # 在图上绘制小红圈


#     plt.axis('off')
#     plt.imshow(original_image_rgba)
#     plt.imshow(color, alpha=0.6)
#     for i in range(len(non_zero_indices)):
#         print("non_zero_indices[i][1]:", non_zero_indices[i][1])
#         print("H - non_zero_indices[i][0]:", H - non_zero_indices[i][0])
#         plt.scatter(non_zero_indices[i][1] , non_zero_indices[i][0] , color='red', s=50, edgecolor='black')  # 在每个点位绘制红圈，s=50 为圈的大小
#         # plt.scatter(non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='blue', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='green', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][1] , H - non_zero_indices[i][0] , color='yellow', s=50, edgecolor='black')
    
    
#     plt.savefig(save_path+"_original.jpg", bbox_inches='tight')
#     plt.close() 
    
    
#     fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     plt.xlim(0, W)
#     plt.ylim(H, 0)
#     # figsize 是图像的尺寸，facecolor 是图像的背景颜色，frameon 是是否显示边框

#     plt.axis('off')
#     plt.imshow(GT_img_L)
#     plt.imshow(color, alpha=0.6)
#     for i in range(len(non_zero_indices)):
#         print("non_zero_indices[i][1]:", non_zero_indices[i][1])
#         print("H - non_zero_indices[i][0]:", H - non_zero_indices[i][0])
#         plt.scatter(non_zero_indices[i][1] , non_zero_indices[i][0] , color='red', s=50, edgecolor='black')  # 在每个点位绘制红圈，s=50 为圈的大小
#         # plt.scatter(non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='blue', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='green', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][1] , H - non_zero_indices[i][0] , color='yellow', s=50, edgecolor='black')
#     plt.savefig(save_path+"_GT.jpg", bbox_inches='tight')
#     plt.close() 




#     fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     plt.xlim(0, W)
#     plt.ylim(H, 0)
#     plt.axis('off')
#     ## IMSHOW 函数用于显示图像，mark_boundaries 函数用于在图像上绘制超像素的边界
#     ## mark_boundaries 函数的第一个参数是图像，第二个参数是超像素的索引
#     plt.imshow(original_image_rgba)
#     plt.imshow(color, alpha=0.3)
#     plt.imshow(mark_boundaries(np.zeros_like(original_image_rgba), connected, color=(1, 1, 0)), alpha=0.5)
#     plt.axis('off')
#     plt.savefig(save_path+"_Boundary.jpg", bbox_inches='tight') ## bbox_inches='tight' 是用于去除图像周围的空白区域
#     plt.close()
#     # 在函数结束时注销颜色映射
#     if 'my_colormap' in plt.colormaps():
#         plt.get_cmap('my_colormap').name = None
#         plt.cm.unregister_cmap('my_colormap')
####################################################################################################################################################


    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', rgb)
    mymap.set_bad(alpha=0)
    plt.register_cmap(name='my_colormap', cmap=mymap)
    plt.set_cmap('my_colormap')
    
    norm = mcolors.Normalize(vmin=0, vmax=NUM_CLASSES-1)
    m = cm.ScalarMappable(norm=norm, cmap=mymap)
  
    color = m.to_rgba(propagated)

    original_image_rgb = np.array(original_image.convert("RGB"))
    GT_img_rgb = np.array(GT_img.convert('RGB'))

    non_zero_indices = sparse_labels_to_mask(sparse_labels, H, W)

    # 创建一个1x3的子图布局
    fig, axs = plt.subplots(2, 2, figsize=(40, 40))

    # 1. 原始图像和传播结果
    axs[0, 0].imshow(original_image_rgb)
    axs[0, 0].imshow(color, alpha=0.6)
    for i in range(len(non_zero_indices)):
        axs[0, 0].scatter(non_zero_indices[i][1], non_zero_indices[i][0], color='red', s=50, edgecolor='black')
    axs[0, 0].set_title('Original Image with Propagation', fontsize=20)
    axs[0, 0].axis('off')

    # 2. GT图像和传播结果
    axs[0, 1].imshow(GT_img_rgb)
    axs[0, 1].imshow(color, alpha=0.6)
    for i in range(len(non_zero_indices)):
        axs[0, 1].scatter(non_zero_indices[i][1], non_zero_indices[i][0], color='red', s=50, edgecolor='black')
    axs[0, 1].set_title('Ground Truth with Propagation', fontsize=20)
    axs[0, 1].axis('off')

    # 3. 原始图像、传播结果和超像素边界
    axs[1, 0].imshow(original_image_rgb)
    axs[1, 0].imshow(color, alpha=0.3)
    axs[1, 0].imshow(mark_boundaries(np.zeros_like(original_image_rgb), connected, color=(1, 1, 0)), alpha=0.5)
    axs[1, 0].set_title('Original Image with Propagation and Boundaries', fontsize=20)
    axs[1, 0].axis('off')

    # 4. 原始图像
    axs[1, 1].imshow(original_image_rgb)
    axs[1, 1].set_title('Original Image', fontsize=20)
    axs[1, 1].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path + "_" + str(num_labels) + "_combined.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    # 在函数结束时注销颜色映射
    if 'my_colormap' in plt.colormaps():
        plt.get_cmap('my_colormap').name = None
        plt.cm.unregister_cmap('my_colormap')



# 这个函数用于创建增强后的地面真实标签的RGB输出
def plot_propagated_json(NUM_CLASSES, save_path, propagated, original_image, sparse_labels, H, W, connected, num_labels):
    ####### 函数用于将传播后的标签绘制为RGB图像 ########
    # 假设传播已由prop_to_unlabelled_spix_feat函数完成

    if NUM_CLASSES == 35:
        # UCSD Mosaics数据集的颜色映射
        # colors = [[167, 18, 159], [180, 27, 92], ..., [131, 69, 63]]  # 35个类别的颜色列表
        colors = [[123, 45, 67],
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
                [255, 0, 255],
                [128, 128, 128],
                [255, 192, 203],
                [255, 165, 0],
                [75, 0, 130],
                [255, 20, 147],
                [0, 128, 128],
                [128, 0, 128],
                [255, 105, 180],
                [0, 0, 0],
                [255, 255, 255],
                [128, 0, 0],
                [0, 128, 0],
                [0, 0, 128],
                [128, 128, 0],
                [0, 128, 128],
                [192, 192, 192],
                [255, 140, 0],
                [255, 69, 0],
                [255, 228, 196],
                [255, 228, 181],
                [255, 222, 173],
                [255, 160, 122],
                [255, 99, 71],
                [255, 127, 80],
                [255, 228, 225],
                [240, 230, 140],
                [255, 228, 196],
                [255, 218, 185],
                [255, 228, 181],
                [255, 240, 245]
        ]

        bgr = np.array(colors) / 255.
        rgb = bgr[:,::-1]  # 将BGR转换为RGB

    elif NUM_CLASSES == 12:
        # CSIRO Segmentation数据集的颜色映射
        colors = [[0, 0, 0], [255, 0, 0], ..., [0, 0, 255]]  # 12个类别的颜色列表
        rgb = np.array(colors) / 255.
    elif NUM_CLASSES == 100:
        colors = {"NON_MIL": [0, 0, 0], "SSID": [30, 60, 90], "CAL_CCA_DC": [60, 120, 180], "NON_FREE": [90, 180, 14], "MALG": [120, 240, 104], "Dict": [150, 44, 194], "MASE_SMO_P": [180, 104, 28], "ACR-HIP": [210, 164, 118], "MASE_LRG_O": [240, 224, 208], "LSUB_SEDI": [14, 28, 42], "EAM_Sub": [44, 88, 132], "POR-MASS": [74, 148, 222], "OTH-SF": [104, 208, 56], "IRCI": [134, 12, 146], "SINV_SFC_O": [164, 72, 236], "PSEU": [194, 132, 70], "MACR_Cal_H": [224, 192, 160], "BRA_ARB_Ac": [254, 252, 250], "B_Monti": [28, 56, 84], "CVIR": [58, 116, 174], "ENSP": [88, 176, 8], "AGAR": [118, 236, 98], "SINV_SPO_E": [148, 40, 188], "DLAB": [178, 100, 22], "POR_NOD": [208, 160, 112], "SGRASS": [238, 220, 202], "Sand": [12, 24, 36], "SpMass": [42, 84, 126], "DSUB": [72, 144, 216], "BRA_FIN_Se": [102, 204, 50], "POR_Com_fi": [132, 8, 140], "POCI_CAU": [162, 68, 230], "BRA_TAB-Ac": [192, 128, 64], "Sediment": [222, 188, 154], "OCOM": [252, 248, 244], "FAV-MUS": [26, 52, 78], "MON_Cap_br": [56, 112, 168], "UTEN": [86, 172, 2], "EAM_RB": [116, 232, 92], "GORG": [146, 36, 182], "ACR-PE": [176, 96, 16], "ERHD": [206, 156, 106], "MASE_SML_O": [236, 216, 196], "POR-ENC": [10, 20, 30], "MLAG": [40, 80, 120], "Lvar": [70, 140, 210], "OTH-SINV": [100, 200, 44], "MACR_Cal_P": [130, 4, 134], "MASE_MEA_L": [160, 64, 224], "ALC-SF": [190, 124, 58], "POR-BRA": [220, 184, 148], "BRA_DIG_Ac": [250, 244, 238], "MPATU": [24, 48, 72], "ENGR1": [54, 108, 162], "Turfsa": [84, 168, 252], "TFP_RDG_Al": [114, 228, 86], "PPOR": [144, 32, 176], "Turf": [174, 92, 10], "SINV_SFC_A": [204, 152, 100], "POCI": [234, 212, 190], "SCplu": [8, 16, 24], "MACR_Fol_P": [38, 76, 114], "CCA": [68, 136, 204], "SINV_SPO_M": [98, 196, 38], "Unc": [128, 0, 128], "ZOAN": [158, 60, 218], "FISH": [188, 120, 52], "MASE_LRG_I": [218, 180, 142], "EAM_DHC": [248, 240, 232], "Mille": [22, 44, 66], "LSUB_SAND": [52, 104, 156], "OTH-HC": [82, 164, 246], "CYAN": [112, 224, 80], "ROSP": [142, 28, 170], "SINV_HYD": [172, 88, 4], "AMAT": [202, 148, 94], "BRA_SMO_Po": [232, 208, 184], "MINV_Dia": [6, 12, 18], "ACR-TCD": [36, 72, 108], "BRA_VER_Po": [66, 132, 198], "MADR": [96, 192, 32], "PASTR": [126, 252, 122], "ACR-OTH": [156, 56, 212], "ACR-BRA": [186, 116, 46], "MON_Cap_pl": [216, 176, 136], "ERGR": [246, 236, 226], "MASE_MEA_O": [20, 40, 60], "BRA_RND_St": [50, 100, 150]}
    elif NUM_CLASSES == 256:
        colors = color_map_256
        rgb = np.array(colors) / 255.
    else:
        print("我们没有存储该类别数量的颜色映射，请指定 - 请查看plot_propagated函数。")



# ############################################################################################################
#     mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', rgb)

#     mymap.set_bad(alpha=0)  # 设置颜色映射如何处理"坏"值
#     plt.register_cmap(name='my_colormap', cmap=mymap)

#     plt.set_cmap('my_colormap')
    
#     # 设置颜色映射的归一化范围
#     norm = mcolors.Normalize(vmin=0, vmax=NUM_CLASSES-1)
#     m = cm.ScalarMappable(norm=norm, cmap=mymap)
  
#     # 将传播后的标签转换为颜色
#     ## color_mask 的形状为 [H, W, 4]
#     color = m.to_rgba(propagated)

#     # 将原图转换为RGBA格式
#     original_image_rgba = np.array(original_image.convert("RGB"))
#     GT_img_L = np.array(GT_img.convert('RGB'))

#     # # 创建一个透明掩码
#     # alpha_mask = color_mask[..., 3]  # 获取颜色掩码的alpha通道
#     # color_mask[..., 3] = alpha_mask * 0.5  # 设置掩码的透明度

#     # # 合成原图和颜色掩码
#     # # color_mask[..., 3:] 的形状为 [H, W, 1]
#     # # color_mask[..., :3] 的形状为 [H, W, 3]
#     # # 1 - color_mask[..., 3:] 的形状为 [H, W, 1]
#     # # color_mask[..., :3] * color_mask[..., 3:] 的形状为 [H, W, 3]
#     # combined_image = (0.8) * original_image_rgba + color_mask * 0.95

#     # # 创建并保存图像
#     # fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     # plt.axis('off')
#     # plt.imshow(combined_image.astype(np.uint8))
#     # plt.savefig(save_path+".jpg", bbox_inches='tight')
#     # plt.close()
#     non_zero_indices = sparse_labels_to_mask(sparse_labels, H, W)

#     fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     plt.xlim(0, W)
#     plt.ylim(H, 0)
#     # figsize 是图像的尺寸，facecolor 是图像的背景颜色，frameon 是是否显示边框
    
#     # 在图上绘制小红圈


#     plt.axis('off')
#     plt.imshow(original_image_rgba)
#     plt.imshow(color, alpha=0.6)
#     for i in range(len(non_zero_indices)):
#         print("non_zero_indices[i][1]:", non_zero_indices[i][1])
#         print("H - non_zero_indices[i][0]:", H - non_zero_indices[i][0])
#         plt.scatter(non_zero_indices[i][1] , non_zero_indices[i][0] , color='red', s=50, edgecolor='black')  # 在每个点位绘制红圈，s=50 为圈的大小
#         # plt.scatter(non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='blue', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='green', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][1] , H - non_zero_indices[i][0] , color='yellow', s=50, edgecolor='black')
    
    
#     plt.savefig(save_path+"_original.jpg", bbox_inches='tight')
#     plt.close() 
    
    
#     fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     plt.xlim(0, W)
#     plt.ylim(H, 0)
#     # figsize 是图像的尺寸，facecolor 是图像的背景颜色，frameon 是是否显示边框

#     plt.axis('off')
#     plt.imshow(GT_img_L)
#     plt.imshow(color, alpha=0.6)
#     for i in range(len(non_zero_indices)):
#         print("non_zero_indices[i][1]:", non_zero_indices[i][1])
#         print("H - non_zero_indices[i][0]:", H - non_zero_indices[i][0])
#         plt.scatter(non_zero_indices[i][1] , non_zero_indices[i][0] , color='red', s=50, edgecolor='black')  # 在每个点位绘制红圈，s=50 为圈的大小
#         # plt.scatter(non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='blue', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][0] , H - non_zero_indices[i][1] , color='green', s=50, edgecolor='black')
#         # plt.scatter(H - non_zero_indices[i][1] , H - non_zero_indices[i][0] , color='yellow', s=50, edgecolor='black')
#     plt.savefig(save_path+"_GT.jpg", bbox_inches='tight')
#     plt.close() 




#     fig = plt.figure(figsize=(20,20), facecolor='w', frameon=False)
#     plt.xlim(0, W)
#     plt.ylim(H, 0)
#     plt.axis('off')
#     ## IMSHOW 函数用于显示图像，mark_boundaries 函数用于在图像上绘制超像素的边界
#     ## mark_boundaries 函数的第一个参数是图像，第二个参数是超像素的索引
#     plt.imshow(original_image_rgba)
#     plt.imshow(color, alpha=0.3)
#     plt.imshow(mark_boundaries(np.zeros_like(original_image_rgba), connected, color=(1, 1, 0)), alpha=0.5)
#     plt.axis('off')
#     plt.savefig(save_path+"_Boundary.jpg", bbox_inches='tight') ## bbox_inches='tight' 是用于去除图像周围的空白区域
#     plt.close()
#     # 在函数结束时注销颜色映射
#     if 'my_colormap' in plt.colormaps():
#         plt.get_cmap('my_colormap').name = None
#         plt.cm.unregister_cmap('my_colormap')
####################################################################################################################################################


    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', rgb)
    mymap.set_bad(alpha=0)
    plt.register_cmap(name='my_colormap', cmap=mymap)
    plt.set_cmap('my_colormap')
    
    norm = mcolors.Normalize(vmin=0, vmax=NUM_CLASSES-1)
    m = cm.ScalarMappable(norm=norm, cmap=mymap)
  
    color = m.to_rgba(propagated)

    original_image_rgb = np.array(original_image.convert("RGB"))

    non_zero_indices = sparse_labels_to_mask(sparse_labels, H, W)

    # 创建一个1x3的子图布局
    fig, axs = plt.subplots(1, 3, figsize=(60, 20))

    # 1. 原始图像和传播结果
    axs[0].imshow(original_image_rgb)
    axs[0].imshow(color, alpha=0.6)
    for i in range(len(non_zero_indices)):
        axs[0].scatter(non_zero_indices[i][1], non_zero_indices[i][0], color='red', s=100, edgecolor='black')
    axs[0].set_title('Original Image with Propagation', fontsize=20)
    axs[0].axis('off')

    # 2. 原始图像、传播结果和超像素边界
    axs[1].imshow(original_image_rgb)
    axs[1].imshow(color, alpha=0.3)
    axs[1].imshow(mark_boundaries(np.zeros_like(original_image_rgb), connected, color=(1, 1, 0)), alpha=0.5)
    axs[1].set_title('Original Image with Propagation and Boundaries', fontsize=20)
    axs[1].axis('off')

    # 3. 原始图像
    axs[2].imshow(original_image_rgb)
    for i in range(len(non_zero_indices)):
        axs[2].scatter(non_zero_indices[i][1], non_zero_indices[i][0], color='red', s=100, edgecolor='black')
    axs[2].set_title('Original Image', fontsize=20)
    axs[2].axis('off')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path + "_" + str(num_labels) + "_combined.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    # 在函数结束时注销颜色映射
    if 'my_colormap' in plt.colormaps():
        plt.get_cmap('my_colormap').name = None
        plt.cm.unregister_cmap('my_colormap')








# 这个函数将最相似超像素的类别传播到没有点标签的超像素
# 我们希望增强的地面真实中的所有像素都有一个关联的标签
def prop_to_unlabelled_spix_feat(sparse_labels, connected, features_cnn, H, W):
    ## sparse_labels 是已标记像素的标签，connected 是超像素的连接信息，features_cnn 是CNN特征，H 和 W 是图像的高度和宽度

    ##### 函数用于将已标记超像素的标签传播到图像中未标记的超像素 #####

    # 将CNN特征转换为NumPy数组并重塑
    features_cnn = features_cnn.detach().cpu().numpy()      # 形状 = [B, N, C]
    features_cnn = features_cnn[0]                          # 形状 = [N, C]
    features_cnn_reshape = np.reshape(features_cnn, (H,W, np.shape(features_cnn)[1]))       # 形状 = [H, W, C]

    spix_features = []

    # 为每个连通的簇找到平均特征向量（我们不再使用优化器中的相同簇）
    # 遍历每个超像素并平均该区域的特征
    for spix in np.unique(connected): ## unique 返回数组中唯一的元素, 返回的是一个数组。因此这行代码是在遍历每个超像素
        r, c = np.where(connected == spix)          
        features_curr_spix = features_cnn_reshape[(r,c)]    # 形状 = [X, C]，其中X是'spix'超像素中的像素数
        average_features = np.mean(features_curr_spix, axis=0, keepdims=True)      # 形状 = [1, C] ## 计算每个超像素(中所有点)的平均特征
        average_features = np.squeeze(average_features)     # 形状 = [C,]
        temp = [spix]                                       # 形状 = [1,]  - 这是当前超像素的索引
        temp.extend(average_features)                       # 形状 = [C+1]  - 我们将超像素索引作为第一个值，然后连接C个特征 ## extend 函数用于在列表末尾添加一个元素
        spix_features.append(temp)

    # 包含所有超像素及其平均特征向量的数组
    spix_features = np.array(spix_features)                 # 形状 = [K_new, C+1]  - 对于每个连通的超像素（可能与指定的K不同），我们有索引和特征

    mask_np = np.array(sparse_labels)
    mask_np = np.squeeze(mask_np)                           # 形状 = [H, W]

    labels = []
    image_size = np.shape(mask_np)
    # 遍历掩码中的每个像素
    for y in range(image_size[0]):
        for x in range(image_size[1]):
            if mask_np[y,x]>0:
                spixel_num = connected[int(y), int(x)]
                labels.append([mask_np[y,x]-1, spixel_num, y, x]) ## 这是类别！
    
    # 包含已标记像素的数组 - 对于每个像素，我们有标签号、它所在的超像素索引和随机点的x,y坐标
    labels_array = np.array(labels)                         # 形状 = [num_points, 4]，num_points 就是 H*W 

    ##! 注意，label与spix不是同一个东西，label是类别，spix是超像素索引

    spix_labels = []
    # 遍历图像中的超像素
    for spix_i in range(len(np.unique(connected))):
        # 如果该超像素已经被标记，那么让我们将其添加到已标记超像素的列表中
        spix = np.unique(connected)[spix_i] ## 获取当前超像素的索引
        if spix in labels_array[:,1]: ## 形状 = [num_points, 4]的参数的第二列，就是在遍历超像素索引
            ## spix 在labels_array 的第二列中
            label_indices = np.where(labels_array[:,1] == spix) ## 获取当前超像素的索引
            labels = labels_array[label_indices]
            most_common = np.argmax(np.bincount(labels[:,0])) ## bincount 返回每个元素出现的次数，然后 argmax 返回出现次数最多的元素的索引


            temp = [spix, most_common] ## [2]
            temp.extend(spix_features[spix_i,1:]) ## extend 函数用于在列表末尾添加一个元素, 形状为[C]
            #! 综合上下两行，temp 的形状为 [C+2]


            spix_labels.append(temp)

    # 创建我们已标记超像素的列表
    spix_labels = np.array(spix_labels)                 #todo 形状 = [K_new_labelled, C+1+1]，其中K_new_labelled 是已标记的超像素数量 - 这个数组只包含已标记的超像素，并指定索引（1）、多数标签（1）和平均特征（C）(上面两行)

    # 创建我们的空传播掩码，准备用每个像素的类别标签填充
    prop_mask = np.empty((image_size[0], image_size[1],)) * np.nan             # 形状 = [H, W]

    #？ 现在再次遍历所有超像素，传播已知和未知的超像素（从上面得到的数组转换到prop_mask中）
    for spix_i in range(len(np.unique(connected))):
        spix = np.unique(connected)[spix_i]
        # 如果超像素已经被标记，那么在我们的prop_mask中传播该标签
        if spix in spix_labels[:,0]: ## 形状 = [K_new_labelled, C+1+1] 的参数的第0列，就是在遍历超像素索引.
            ## 遍历当前选中的超像素索引下的所有点
            r, c = np.where(connected == spix)  # 获取选定超像素的索引（点的坐标）
            loc = np.where(spix_labels[:,0] == spix) ## 形状 = [K_new_labelled, C+1+1] 的参数的第0列，就是在遍历超像素索引
            class_label = spix_labels[loc][0][1] ## spix_labels[loc]代表的是当前超像素索引对应的类别，[0]代表的是当前超像素索引对应的类别，[1]代表的是类别
            prop_mask[(r,c)] = class_label ## 将类别赋值给prop_mask
        # 如果超像素没有标签，我们需要找到特征最相似的已标记超像素      
        else:
            r, c = np.where(connected == spix)  # 获取选定超像素的索引
            labelled_spix_features = spix_labels[:,2:]               # 形状 = [K_new_labelled, C]
            one_spix_features = spix_features[spix_i,1:]              # 形状 = [C]
            euc_dists = [np.linalg.norm(i-one_spix_features) for i in labelled_spix_features]
            most_similiar_labelled_spix = np.argmin(np.array(euc_dists))              # 形状 = 具有最相似特征的超像素索引的整数
            most_similiar_class_label = spix_labels[most_similiar_labelled_spix][1]    # 形状 = 该超像素对应类别的整数
            prop_mask[(r,c)] = most_similiar_class_label ## 将类别赋值给prop_mask
    #     # 可视化超像素区域
    # plt.figure(figsize=(10, 10))
    # ## IMSHOW 函数用于显示图像，mark_boundaries 函数用于在图像上绘制超像素的边界
    # ## mark_boundaries 函数的第一个参数是图像，第二个参数是超像素的索引
    # plt.imshow(mark_boundaries(np.zeros_like(connected), connected))
    # plt.title('Superpixel Boundaries')
    # plt.axis('off')
    # plt.savefig('./save/superpixel_boundaries.png')
    # plt.close()

    # # 可视化传播后的标签
    # plt.figure(figsize=(10, 10))
    # plt.imshow(prop_mask, cmap='viridis')
    # plt.title('Propagated Labels')
    # plt.colorbar()
    # plt.axis('off')
    # plt.savefig('./save/propagated_labels.png')
    # plt.close()


    return prop_mask, connected
#####################################################.
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input specifications for generating augmented ground truth from randomly distributed point labels.')

    # Paths - these are required
    parser.add_argument('-r', '-read_im', action='store', type=str, dest='read_im', help='the path to the images', required=True, default='./img') # dest是参数的名称
    parser.add_argument('-g', '-read_gt', action='store', type=str, dest='read_gt', help='the path to the provided labels', required=True, default='./img_ann')
    parser.add_argument('-json', '-read_gt_json', action='store', dest='json_path', type=str, default = False)

    parser.add_argument('-l', '-save_labels', action='store', type=str, dest='save_labels', help='the destination of your propagated labels', required=True, default='./save')
    parser.add_argument('-p', '--save_rgb', action='store', type=str, dest='save_rgb', help='the destination of your RGB propagated labels', default='./save')

    # Flags to specify functionality
    parser.add_argument('--ensemble', action='store_true', dest='ensemble', help='use this flag when you would like to use an ensemble of 3 classifiers, otherwise the default is to use a single classifier')
    parser.add_argument('--points', action='store_true', default=False, dest='points', help='use this flag when your labels are already sparse, otherwise the default is dense')
    parser.add_argument('--device', default='gpu')


    # Optional parameters
    # Default values correspond to the UCSD Mosaics dataset
    parser.add_argument('-x', '--xysigma', action='store', type=float, default=0.631, dest='xysigma', help='if NOT using ensemble and if you want to specify the sigma value for the xy component')
    parser.add_argument('-f', '--cnnsigma', action='store', type=float, default=0.5534, dest='cnnsigma', help='if NOT using ensemble and if you want to specify the sigma value for the cnn component')
    parser.add_argument('-a', '--alpha', action='store', type=float, default=1140, dest='alpha', help='if NOT using ensemble and if you want to specify the alpha value for weighting the conflict loss')
    parser.add_argument('-n', '--num_labels', action='store', type=int, default=300, dest='num_labels', help='if labels are dense, specify how many random point labels you would like to use, default is 300')
    parser.add_argument('-y', '--height', action='store', type=int, default=512, dest='image_height', help='height in pixels of images')
    parser.add_argument('-w', '--width', action='store', type=int, default=512, dest='image_width', help='width in pixels of images')
    parser.add_argument('-c', '--num_classes', action='store', type=int, default=256, dest='num_classes', help='the number of classes in the dataset')
    parser.add_argument('-u', '--unlabeled', action='store', type=int, default=34, dest='unlabeled', help='the index of the unlabeled/unknown/background class')

    args = parser.parse_args()

    # 从命令行参数中提取读取图像、读取标签、保存标签和保存RGB图像的路径
    read_im = args.read_im
    read_gt = args.read_gt
    save_labels = args.save_labels
    save_rgb = args.save_rgb

    # 检查是否使用集成模式和点模式
    ensemble = args.ensemble
    points = args.points

    # 提取sigma_xy、sigma_cnn和alpha的值
    # sigma_xy是用于XY特征的sigma值，用于计算XY特征之间的距离
    sigma_xy = args.xysigma
    # sigma_cnn是用于CNN特征的sigma值，用于计算CNN特征之间的距离
    sigma_cnn = args.cnnsigma
    # alpha是用于调整冲突损失的权重，影响最终损失函数的计算
    alpha = args.alpha

    json_path = args.json_path


    # 获取随机点标签的数量
    num_labels = args.num_labels

    # 获取图像的高度和宽度
    image_height = args.image_height
    image_width = args.image_width

    # 获取未标记类的索引
    unlabeled = args.unlabeled

    if ensemble:
        # 在消融研究中，我们发现了以下三个模型的参数设置，这些值可以根据需要进行调整
        # 模型1的参数设置
        sigma_xy_1 = 0.5597  # XY特征的sigma值，用于计算XY特征之间的距离
        sigma_cnn_1 = 0.5539  # CNN特征的sigma值，用于计算CNN特征之间的距离
        alpha_1 = 1500  # 调整冲突损失的权重，影响最终损失函数的计算

        # 模型2的参数设置
        sigma_xy_2 = 0.5309
        sigma_cnn_2 = 0.846
        alpha_2 = 1590

        # 模型3的参数设置
        sigma_xy_3 = 0.631 
        sigma_cnn_3 = 0.5534
        alpha_3 = 1140
    else:
        # 如果不使用集成模式，则使用命令行参数指定的sigma_xy、sigma_cnn和alpha值
        sigma_xy = args.xysigma
        sigma_cnn = args.cnnsigma
        alpha = args.alpha

    print("received your values, setting some things up...")



    # 用于计算失真损失的像素数量，以提高速度和降低内存使用量
    num_pixels_used = 3000

    # 检查是否有可用的CUDA设备，如果有则使用CUDA，否则使用CPU
    device = 'cuda' if args.device == 'gpu' else 'cpu'


    # 获取已完成的图像列表
    images_done = os.listdir(save_labels)
    # 获取待处理的图像列表
    images = os.listdir(read_im)

    # 如果脚本在生成过程中被杀死，这将允许在不重复图像的情况下重新启动
    images_filtered = [y for y in images if y not in images_done]

    #.获取类别数量
    NUM_CLASSES = args.num_classes
    # NUM_CLASSES = 100

    # 这是超像素的数量
    k = 100
    # 初始化时的高度和宽度方向的超像素数量
    # 初始化是一个网格，所以如果图像是正方形，我们设置为10x10
    # 如果图像不是正方形，超像素应该根据需要进行调整
    if image_height == image_width:
        k_w = 10
        k_h = 10
    else:
        k_w = 12
        k_h = 8

    learning_rate = 0.1
    num_iterations = 50

    C = 100 # 通常情况下，这个值设置为20个特征
    in_channels = 5
    out_channels = 64

    norm_val_x = k_w/image_width
    norm_val_y = k_h/image_height

    # 获取图像中每个像素的特征
    # xylab参数中的1.0是默认的sigma值，norm_val_x和norm_val_y是归一化因子，用于调整特征的尺度
    xylab_function = xylab(1.0, norm_val_x, norm_val_y)
    CNN_function = CNN(in_channels, out_channels, C) 

    model_dict = CNN_function.state_dict()
    ckp_path = "standardization_C=100_step70000.pth" # 在UCSD上训练，但应用了标准化
    obj = torch.load(ckp_path, map_location=torch.device(device))
    pretrained_dict = obj['net']
    # 1. 过滤掉不必要的键
    # 注意：在预训练模型中，所有参数都有"CNN."在键名前面，这意味着它们不会与加载的CNN匹配（当不加载整个SSN时）
    pretrained_dict = {key[4:]: val for key, val in pretrained_dict.items() if key[4:] in model_dict}  
    # 2. 覆盖现有状态字典中的条目
    model_dict.update(pretrained_dict) 
    CNN_function.load_state_dict(pretrained_dict)
    CNN_function.to(device)
    CNN_function.eval()

    #。 根据网格初始化计算每个超像素的平均特征（质心）
    spixel_centres = get_spixel_init(k, image_width, image_height)

    # We only need to calculate metrics if we have dense ground truth
    if points == False:
        # pa_metric = torchmetrics.Accuracy(num_classes = NUM_CLASSES, ignore_index=unlabeled)
        # mpa_metric = torchmetrics.Accuracy(num_classes = NUM_CLASSES, ignore_index=unlabeled, average='macro')
        # iou_metric = torchmetrics.JaccardIndex(num_classes = NUM_CLASSES, ignore_index=unlabeled, reduction='none')
        # 使用torchmetrics库计算准确率（Accuracy）、宏平均准确率（Macro Average Accuracy）和交并比（IoU）
        pa_metric = torchmetrics.Accuracy(num_classes=NUM_CLASSES, ignore_index=unlabeled, task='multiclass')
        mpa_metric = torchmetrics.Accuracy(num_classes=NUM_CLASSES, ignore_index=unlabeled, average='macro', task='multiclass')
        iou_metric = torchmetrics.JaccardIndex(num_classes=NUM_CLASSES, ignore_index=unlabeled, task='multiclass')

    print("setup is complete, now iterating through your images...")

    ### Iterate through the specified images ###
    for image_name in images_filtered:

        if json_path:
            json_realname = image_name[:-4] + ".json"
            json_name = os.path.join(json_path,json_realname)
            with open(json_name, 'r') as f:
                gt_json = json.load(f)
            pil_img = Image.open(os.path.join(read_im,image_name)).convert('RGB')  #.resize((image_width, image_height)
        else:
            pil_img = Image.open(os.path.join(read_im,image_name)).convert('RGB')  #.resize((image_width, image_height)
            GT_pil_img = Image.open(os.path.join(read_gt,image_name)).convert('L')  # .resize((image_width, image_height), Image.NEAREST
             # 将地面真值PIL图像转换为NumPy数组
            GT_mask_np = np.array(GT_pil_img)
            
            # 将NumPy数组转换为PyTorch张量
            GT_mask = torch.from_numpy(GT_mask_np)
            
            # 在最后一个维度上增加一个维度，将2D掩码转换为3D
            GT_mask_torch = np.expand_dims(GT_mask, axis=2)
            
            # 创建一个转换管道，将NumPy数组转换为PyTorch张量
            transform = transforms.Compose([ToTensor()])
            
            # 应用转换，将3D NumPy数组转换为PyTorch张量
            # 这个操作会将数据范围从[0, 255]缩放到[0.0, 1.0]，并调整维度顺序
            GT_mask_torch = transform(GT_mask_torch)
#????????????????????????????????????????????????????????????????????????
        # pil_img = Image.open(os.path.join(read_im,image_name))  #.resize((image_width, image_height)
        # GT_pil_img = Image.open(os.path.join(read_gt,image_name))  # .resize((image_width, image_height), Image.NEAREST
#????????????????????????????????????????????????????????????????????????


        # 将PIL图像转换为NumPy数组
        image = np.array(pil_img)

        # # 如果我们有密集的地面真值掩码，我们需要选择num_labels个像素进行传播
        # if points == False:
        #     # 在地面真值掩码中随机选择一个标记点的子集：
        #     # 创建一个与图像大小相同的零矩阵
        #     sparse_mask = np.zeros(image_height*image_width, dtype=int)
        #     # 将前num_labels个元素设置为1，表示这些位置将被选为标记点
        #     sparse_mask[:num_labels] = 1
        #     # 随机打乱数组，确保标记点随机分布
        #     np.random.shuffle(sparse_mask)
        #     # 将一维数组重塑为与图像相同的二维形状
        #     sparse_mask = np.reshape(sparse_mask, (image_height, image_width))
        #     # 在第一个维度上增加一个维度，为批处理准备
        #     sparse_mask = np.expand_dims(sparse_mask, axis=0) # [1, H, W]

        #     # 我们给所有类别加1，这样'0'就成为所有未标记的像素
        #     # 将地面真值掩码的所有值加1
        #     # sparse_labels = torch.add(GT_mask_torch, 1)
        #     sparse_labels = GT_mask_torch.to(torch.int32) # [1, H, W]
        #     sparse_labels = sparse_labels - 1
        #     # 将加1后的掩码与sparse_mask相乘，只保留随机选择的点
        #     sparse_labels = sparse_labels * sparse_mask
        #     # 在第一个维度上增加一个维度，并将张量移动到指定设备上
        #     sparse_labels = torch.unsqueeze(sparse_labels, 0).to(device)  # 形状为 [B, 1, H, W]

        #     '''
        #     总结：随机选取num_labels个像素，将这些像素的标签设置为1，其他像素的标签设置为0,然后通过操作（先让gt—mask全员加一，再逐点相乘），可以过滤掉未被选中的点
        #     最后，再在第一个维度上增加一个维度，并将张量移动到指定设备上，进行调整维度等操作
        #     '''

#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

        # 如果我们有密集的地面真值掩码，我们需要选择num_labels个像素进行传播
        if points == False:
            # 将 GT_mask_torch 转换为 numpy 数组
            GT_mask_np = GT_mask_torch.squeeze().cpu().numpy()
            
            # 找到所有非零像素的索引
            non_zero_indices = np.argwhere(GT_mask_np > 0)
            
            # 如果非零像素数量少于 num_labels，则使用所有非零像素
            if len(non_zero_indices) <= num_labels:
                selected_indices = non_zero_indices
            else:
                # 随机选择 num_labels 个非零像素
                # np.random.choice(len(non_zero_indices), num_labels, replace=False) 从非零像素中随机选择 num_labels 个像素，replace=False 表示不重复选择
                # len(non_zero_indices) 是数组的长度，num_labels 是我们要选择的像素数量
                selected_indices = non_zero_indices[np.random.choice(len(non_zero_indices), num_labels, replace=False)]
            
            # 创建一个与原图像大小相同的零矩阵
            sparse_mask = np.zeros_like(GT_mask_np, dtype=int)
            
            # 将选中的像素在 sparse_mask 中设置为 1
            sparse_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
            
            # 将 numpy 数组转换回 PyTorch 张量
            sparse_mask = torch.from_numpy(sparse_mask).unsqueeze(0).unsqueeze(0)  # 形状为 [1, 1, H, W]
            
            # 创建 sparse_labels
            sparse_labels = GT_mask_torch.clone()
            # sparse_labels[sparse_labels > 0] -= 1  # 将类别标签减 1，使背景为 0
            sparse_labels = sparse_labels * sparse_mask
            
            # 将张量移动到指定设备上
            sparse_labels = sparse_labels.to(device)  # 形状为 [1, 1, H, W]



#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？



        # We are provided with randomly distributed points:
        elif not json_path:
            sparse_labels = torch.unsqueeze(GT_mask_torch, 0).to(device)                # shape = [B, 1, H, W]
        elif json_path:
            sparse_labels = torch.zeros(image_height, image_width, dtype=torch.int64).to(device)
            for item in gt_json:
                sparse_labels[item[0], item[1]] = int(item[2])
            sparse_labels = sparse_labels.unsqueeze(0).unsqueeze(0)


        means, stds = find_mean_std(image) # 获取RGB三通道的参数
        image = (image - means) / stds    # shape: [H, W, C] where C is RGB in range [0,255] BUT colour channels are now standardized
        transform = transforms.Compose([img2lab(), ToTensor()]) # transforms.Compose作用为：将两个变换线性连接
        img_lab = transform(image) #。将img转换到Lab空间并转换为tensor
        img_lab = torch.unsqueeze(img_lab, 0) # unsqueeze 增加一个维度

        image_shape = img_lab.shape                                                     # shape = [B, 3, H, W]   where 3 = RGB

        w = image_shape[3]
        h = image_shape[2]

        B = img_lab.shape[0]
        XYLab, X, Y, Lab = xylab_function(img_lab)                                     # shape = [B, 5, H, W]  where 5 = x,y,L,A,B 
        XYLab = XYLab.to(device)
        # XYLab是将Lab通道与XY坐标连接在一起的特征图，作用是：将图像的每个像素的坐标和颜色信息结合在一起，形成一个包含位置和颜色信息的特征图
        X = X.to(device)
        Y = Y.to(device)

        # send the XYLab features through the CNN to obtain the encoded features 
        with torch.no_grad():
            features = CNN_function(XYLab)                                             # shape = [B, C, H, W]  where C = 20 from config file   

        features_magnitude_mean = torch.mean(torch.norm(features, p=2, dim=1))
        # features_rescaled 的含义是
        features_rescaled = (features / features_magnitude_mean)
        features_cat = torch.cat((X, Y, features_rescaled), dim = 1)
        XY_cat = torch.cat((X, Y), dim = 1)
        
        mean_init = compute_init_spixel_feat(features_cat, torch.from_numpy(spixel_centres[0].flatten()).long().to(device), k)   # shape = [B, K, C] , C = 5，有5个通道的特征
        # spixel_centres[0]是因为spixel_centres是一个列表，列表的第一个元素是初始化的超像素图

        # 将features_rescaled从[B, C, H, W]展平为[B, C, N]，其中N = H * W
        # B是批次大小，C是特征通道数（这里应为15），N是像素总数
        CNN_features = torch.flatten(features_rescaled, start_dim=2, end_dim=3)       # 形状 = [B, C, N]，这里C应为15
        # 交换最后两个维度，将形状从[B, C, N]变为[B, N, C]
        CNN_features = torch.transpose(CNN_features, 2, 1)                            # 形状 = [B, N, C]

        # 将XY_cat从[B, C, H, W]展平为[B, C, N]，其中N = H * W
        # B是批次大小，C是特征通道数（这里应为2，表示X和Y坐标），N是像素总数
        XY_features = torch.flatten(XY_cat, start_dim=2, end_dim=3)                   # 形状 = [B, C, N]，这里C应为2
        # 交换最后两个维度，将形状从[B, C, N]变为[B, N, C]
        XY_features = torch.transpose(XY_features, 2, 1)                              # 形状 = [B, N, C]

        features_cat = torch.flatten(features_cat, start_dim=2, end_dim=3)            # shape = [B, C, N] but here we should have C = 17
        features_cat = torch.transpose(features_cat, 2, 1)                            # shape = [B, N, C]

        torch.backends.cudnn.benchmark = True
        
        if ensemble:
            criterion_1 = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_1, sigma_val_cnn=sigma_cnn_1, alpha=alpha_1, num_pixels_used=num_pixels_used).to(device)
            optimizer_1 = Adam(criterion_1.parameters(), lr = learning_rate)
            scheduler_1 = lr_scheduler.ReduceLROnPlateau(optimizer_1, factor=0.1, patience=1, min_lr = 0.0001)

            criterion_2 = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_2, sigma_val_cnn=sigma_cnn_2, alpha=alpha_2, num_pixels_used=num_pixels_used).to(device)
            optimizer_2 = Adam(criterion_2.parameters(), lr = learning_rate)
            scheduler_2 = lr_scheduler.ReduceLROnPlateau(optimizer_2, factor=0.1, patience=1, min_lr = 0.0001)

            criterion_3 = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy_3, sigma_val_cnn=sigma_cnn_3, alpha=alpha_3, num_pixels_used=num_pixels_used).to(device)
            optimizer_3 = Adam(criterion_3.parameters(), lr = learning_rate)
            scheduler_3 = lr_scheduler.ReduceLROnPlateau(optimizer_3, factor=0.1, patience=1, min_lr = 0.0001)

            best_clusters_1 = optimize_spix(criterion_1, optimizer_1, scheduler_1,  norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members_1 = members_from_clusters(sigma_xy_1, sigma_cnn_1, XY_features, CNN_features, best_clusters_1)

            best_clusters_2 = optimize_spix(criterion_2, optimizer_2, scheduler_2, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members_2 = members_from_clusters(sigma_xy_2, sigma_cnn_2, XY_features, CNN_features, best_clusters_2)

            best_clusters_3 = optimize_spix(criterion_3, optimizer_3, scheduler_3, norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)
            best_members_3 = members_from_clusters(sigma_xy_3, sigma_cnn_3, XY_features, CNN_features, best_clusters_3)

            # MAJORITY VOTE FROM THE THREE CLASSIFIERS
            best_members_1_max = torch.squeeze(torch.argmax(best_members_1, 2))
            best_members_2_max = torch.squeeze(torch.argmax(best_members_2, 2))
            best_members_3_max = torch.squeeze(torch.argmax(best_members_3, 2))

            # Clear some extra variables from the memory
            del best_members_1, best_members_2, best_members_3

            connected_1 = enforce_connectivity(best_members_1_max, h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]
            connected_2 = enforce_connectivity(best_members_2_max, h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]
            connected_3 = enforce_connectivity(best_members_3_max, h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]

            # If there are unlabelled superpixels, we propagate the class of the superpixel with the most similar features
            prop_1, connected_1 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_1, CNN_features, image_height, image_width)
            prop_2, connected_2 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_2, CNN_features, image_height, image_width)
            prop_3, connected_3 = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected_3, CNN_features, image_height, image_width)

            prop_1_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[prop_1.astype(np.int32)] ## np.eye代表
            prop_2_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[prop_2.astype(np.int32)]
            prop_3_onehot = np.eye(NUM_CLASSES, dtype=np.int32)[prop_3.astype(np.int32)]

            # Add together
            prop_count = prop_1_onehot + prop_2_onehot + prop_3_onehot

            del prop_1_onehot, prop_2_onehot, prop_3_onehot

            # The unlabeled class to be either first (0) or last
            if unlabeled == 0:
                propagated_full = np.argmax(prop_count[:,:,1:], axis=-1) + 1
                propagated_full[prop_count[:,:,0] == 3] = 0
            else:
                propagated_full = np.argmax(prop_count[:,:,:-1], axis=-1)
                propagated_full[prop_count[:,:,unlabeled] == 3] = unlabeled
            
            # 1. 合并三个connected数组
            connected_stack = np.stack([connected_1, connected_2, connected_3], axis=-1)
            connected = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=-1, arr=connected_stack)


        else:
            # Single classifier, so just do everything once
            criterion = CustomLoss(mean_init, w*h, XY_features, CNN_features, features_cat, sparse_labels, sigma_val_xy=sigma_xy, sigma_val_cnn=sigma_cnn, alpha=alpha, num_pixels_used=num_pixels_used).to(device)
            optimizer = Adam(criterion.parameters(), lr = learning_rate)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=1, min_lr = 0.0001)
            best_clusters = optimize_spix(criterion, optimizer, scheduler,  norm_val_x=norm_val_x, norm_val_y=norm_val_y, num_iterations=num_iterations)

            # 使用最优的clusters，计算每个像素的软成员资格
            best_members = members_from_clusters(sigma_xy, sigma_cnn, XY_features, CNN_features, best_clusters)
            # 使用软成员资格，强制连接性，将未标记的像素连接到最近的标记像素
            ## torch.squeeze(torch.argmax(best_members, 2)) 是找出每个像素最可能属于哪个类别，返回的是一个一维数组，形状为[N]，squeeze的作用是去掉一维
            connected = enforce_connectivity(torch.squeeze(torch.argmax(best_members, 2)), h, w, k, connectivity = True)  # connectivity=True normally                       # shape = [H, W]
            # 将未标记的像素传播到最近的标记像素
            propagated_full, connected = prop_to_unlabelled_spix_feat(sparse_labels.detach().cpu(), connected, CNN_features, image_height, image_width)

        # Whether using an ensemble or not, we now have a propagated mask
        # 检查用户是否希望保存掩码的RGB版本，如果是则保存
        if save_rgb is not None:
            if not json_path:
                plot_propagated(NUM_CLASSES, os.path.join(save_rgb, image_name[:-4]), propagated_full, Image.open(os.path.join(read_im,image_name)).convert('RGBA'), sparse_labels, image_height, image_width, Image.open(os.path.join(read_gt,image_name)).convert('L'), connected, num_labels)
            elif json_path:
                plot_propagated_json(NUM_CLASSES, os.path.join(save_rgb, image_name[:-4]), propagated_full, Image.open(os.path.join(read_im,image_name)).convert('RGBA'), sparse_labels, image_height, image_width, connected, num_labels)

        # # 将传播后的掩码保存为PNG文件到指定目录
        # propagated_as_image = Image.fromarray(propagated_full.astype(np.uint8))
        # propagated_as_image.save(os.path.join(save_labels,image_name[:-4])+".png", "PNG")