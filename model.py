import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import *
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    # VGG基础模型，用于产生低维向量
    
    def __init__(self):
        super(VGGBase, self).__init__()

        # VGG16中的标准卷积层
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # ceil_mode=True, 向上取整

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # SSD要求pool5的kernel_size=3, stride=1, padding=1
        # VGG原本的pool5是kernel_size=2, stride=2, 这里改成kernel_size=3, stride=1, padding=1
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # VGG的全连接修改为卷积
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # 空洞卷积
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_layers()

    def forward(self, image):
            """
            前向传播

            :param image: 输入图像，输入维度(N, 3, 300, 300)
            :return: 来自conv4_3和conv7的特征图
            """

            out = F.relu(self.conv1_1(image))
            out = F.relu(self.conv1_2(out))
            out = self.pool1(out) # (N, 64, 150, 150)

            out = F.relu(self.conv2_1(out))
            out = F.relu(self.conv2_2(out))
            out = self.pool2(out) # (N, 128, 75, 75)

            out = F.relu(self.conv3_1(out))
            out = F.relu(self.conv3_2(out))
            out = F.relu(self.conv3_3(out))
            out = self.pool3(out) # (N, 256, 38, 38)

            out = F.relu(self.conv4_1(out))
            out = F.relu(self.conv4_2(out))
            out = F.relu(self.conv4_3(out))
            conv4_3_feats = out
            out = self.pool4(out) # (N, 512, 19, 19)


            out = F.relu(self.conv5_1(out))
            out = F.relu(self.conv5_2(out))
            out = F.relu(self.conv5_3(out))
            out = self.pool5(out) # (N, 512, 19, 19)

            out = F.relu(self.conv6(out))
            out = F.relu(self.conv7(out))
            conv7_feats = out # (N, 1024, 19, 19)

            return conv4_3_feats, conv7_feats
    
    def load_pretrained_layers(self):
            """
            加载预训练的VGG16模型
            """
            # Current state of base
            state_dict = self.state_dict()
            param_names = list(state_dict.keys())

            # Pretrained VGG base
            pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
            pretrained_param_names = list(pretrained_state_dict.keys())

            # Transfer conv. parameters from pretrained model to current model
            for i, param in enumerate(param_names[:-4]):  # excluding conv6 and conv7 parameters
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

            # Convert fc6, fc7 to convolutional layers, and subsample (by decimation) to sizes of conv6 and conv7
            # fc6
            conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
            conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
            state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
            state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
            # fc7
            conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
            conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
            state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
            state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

            # Note: an FC layer of size (K) operating on a flattened version (C*H*W) of a 2D image of size (C, H, W)...
            # ...is equivalent to a convolutional layer with kernel size (H, W), input channels C, output channels K...
            # ...operating on the 2D image of size (C, H, W) without padding

            self.load_state_dict(state_dict)

            print("\nLoaded base model.\n")

class AuxiliaryConvolutions(nn.Module):
    """
    辅助卷积层,用于产生高维特征图
    """

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        # VGGBase之后进入辅助卷积
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 降维, stride>1

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 降维, stride>1

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # 降维,padding=0

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # 降维,padding=0
        
        self.init_conv2d()


    def init_conv2d(self):
        """
        初始化卷积层
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
        
    def forward(self, conv7_feats):
        """
        前向传播
        
        :param conv7_feats: 来自VGGBase的conv7_feats, 维度(N, 1024, 19, 19)
        :return: 来自conv8_2, conv9_2, conv10_2, conv11_2的g高维特征图
        """
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out)) # (N, 512, 10, 10)
        conv8_2_feats = out

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out)) # (N, 256, 5, 5)
        conv9_2_feats = out

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out)) # (N, 256, 3, 3)
        conv10_2_feats = out

        out = F.relu(self.conv11_1(out))
        out = F.relu(self.conv11_2(out)) # (N, 256, 1, 1)
        conv11_2_feats = out

        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):
    """
    通过特征图产生 定位框和类别预测

    定位框的预测结果是编码后的偏移量,utils.cxcy_to_gcxgcy()就是这个编码过程

    类别得分的预测结果是类别概率,即每个类别属于哪个类别的概率,当'background'类的得分最大时,说明该位置没有物体
    """

    def __init__(self, n_classes):
        """
        :param n_classes: 类别数
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # 这里设定了每个特征图上每个位置的prior-boxes数量
        n_boxes = {'conv4_3': 4,
                   'conv7': 6,
                   'conv8_2': 6,
                   'conv9_2': 6,
                   'conv10_2': 4,
                   'conv11_2': 4}
        # 4 prior-boxes意味着每个特征图上每个位置有4个prior-boxes

        # 定位预测卷积,其预测的是关于每个prior-boxes的偏移量
        self.loc_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * 4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * 4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * 4, kernel_size=3, padding=1)

        # 类别预测卷积,其预测结的是预测定位框中的类别
        self.cl_conv4_3 = nn.Conv2d(512, n_boxes['conv4_3'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, n_boxes['conv7'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, n_boxes['conv8_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, n_boxes['conv9_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, n_boxes['conv10_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, n_boxes['conv11_2'] * n_classes, kernel_size=3, padding=1)

        # 初始化参数
        self.init_conv2d()

    def init_conv2d(self):
        """
        初始化卷积层参数.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
        """
        前向传播

        :param conv4_3_feats: conv4_3特征图 (N, 512, 38, 38)
        :param conv7_feats: conv7特征图 (N, 1024, 19, 19)
        :param conv8_2_feats: conv8_2特征图 (N, 512, 10, 10)
        :param conv9_2_feats: conv9_2特征图 (N, 256, 5, 5)
        :param conv10_2_feats: conv10_2特征图 (N, 256, 3, 3)
        :param conv11_2_feats: conv11_2特征图 (N, 256, 1, 1)
        :return: 每张图上的8732个prior-boxes的偏移量, 以及每个prior-boxes对应的定位框中的类别概率
        """
        batch_size = conv4_3_feats.size(0)

        # 预测定位框的偏移
        l_conv4_3 = self.loc_conv4_3(conv4_3_feats)  # (N, 16, 38, 38)
        l_conv4_3 = l_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_conv4_3 = l_conv4_3.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_conv7 = self.loc_conv7(conv7_feats)  # (N, 24, 19, 19)
        l_conv7 = l_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_conv7 = l_conv7.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_conv8_2 = self.loc_conv8_2(conv8_2_feats)  # (N, 24, 10, 10)
        l_conv8_2 = l_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv8_2 = l_conv8_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_conv9_2 = self.loc_conv9_2(conv9_2_feats)  # (N, 24, 5, 5)
        l_conv9_2 = l_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv9_2 = l_conv9_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv10_2 = self.loc_conv10_2(conv10_2_feats)  # (N, 16, 3, 3)
        l_conv10_2 = l_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_conv10_2 = l_conv10_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_conv11_2 = self.loc_conv11_2(conv11_2_feats)  # (N, 16, 1, 1)
        l_conv11_2 = l_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv11_2 = l_conv11_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # 预测定位框中的类别
        c_conv4_3 = self.cl_conv4_3(conv4_3_feats)  # (N, 4 * n_classes, 38, 38)
        c_conv4_3 = c_conv4_3.permute(0, 2, 3, 1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_conv4_3 = c_conv4_3.view(batch_size, -1, self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_conv7 = self.cl_conv7(conv7_feats)  # (N, 6 * n_classes, 19, 19)
        c_conv7 = c_conv7.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_conv7 = c_conv7.view(batch_size, -1, self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_conv8_2 = self.cl_conv8_2(conv8_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv8_2 = c_conv8_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv8_2 = c_conv8_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)

        c_conv9_2 = self.cl_conv9_2(conv9_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv9_2 = c_conv9_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv9_2 = c_conv9_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv10_2 = self.cl_conv10_2(conv10_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_conv10_2 = c_conv10_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_conv10_2 = c_conv10_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_conv11_2 = self.cl_conv11_2(conv11_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv11_2 = c_conv11_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv11_2 = c_conv11_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # 一共8732个prior-boxes
        # 按照特定顺序把他们拼接起来,其顺序必须和prior-box的顺序一致
        locs = torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2], dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores

class SSD300(nn.Module):
    """
    SSD300模型.
    """

    def __init__(self, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes

        self.base = VGGBase()
        self.aux_convs = AuxiliaryConvolutions()
        self.pred_convs = PredictionConvolutions(n_classes)

        # 由于低维特征图的缩放因子较大,这里使用L2正则化,并且缩放因子是可学习的
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # conv4_3中有512个通道
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        前向传播

        :param image: 图片 (N, 3, 300, 300)
        :return: 每张图上的8732个prior-boxes的偏移量, 以及每个prior-boxes对应的定位框中的类别概率
        """
        # VGGBase 产生低维特征图
        conv4_3_feats, conv7_feats = self.base(image)  # (N, 512, 38, 38), (N, 1024, 19, 19)

        # 对conv4_3进行L2正则化
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3_feats = conv4_3_feats / norm  # (N, 512, 38, 38)
        conv4_3_feats = conv4_3_feats * self.rescale_factors  # (N, 512, 38, 38)

        # 辅助卷积 产生高维特征图
        conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats = \
            self.aux_convs(conv7_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # 预测卷积 产生预测结果
        locs, classes_scores = self.pred_convs(conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats,
                                               conv11_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        创建8732个prior-boxes(默认)

        :return: prior-box的center-size坐标 (8732, 4)
        """
        fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # 对于横纵比为1的prior-boxes, 为其添加一个额外的缩放因子，其大小为当前特征图缩放因子和下一个特征图缩放因子的几何平均值
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # 对于最后一个特征图, 其下一个特征图不存在, 其缩放因子为1
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4); this line has no effect; see Remarks section in tutorial

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        解析预测结果来检测目标.

        对于每个类别, 对大于min_score的prior-boxes进行非极大值抑制.

        :param predicted_locs: 预测结果中的prior-boxes的偏移量 (N, 8732, 4)
        :param predicted_scores: 预测结果中的prior-boxes的类别概率 (N, 8732, n_classes)
        :param min_score: 用于判断一个定位框是否属于某个类别的阈值
        :param max_overlap: 两个定位框之间的最大重叠程度, 其中较小的一个不会被抑制
        :param top_k: 如果所有类中都有大量的检测结果, 则只保留最前面的k个
        :return: 检测结果(boxes, labels, and scores), 列表,其长度为batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # 用于存储检测结果
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # 解码偏移量
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), 分数形式的边界坐标

            # 存储这张图片上的检测结果
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # 仅保留这个类别中大于阈值的prior-boxes和其对应的类别概率
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # bool张量
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # 通过分数来排序定位框和分数
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # 找到筛选后的定位框之间的交并比
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # 一个用于保存哪些box需要被抑制的布尔张量
                # True表示抑制, False表示不抑制
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)  # (n_qualified)

                # 按得分降序遍历所有筛选后的定位框
                for box in range(class_decoded_locs.size(0)):
                    # 如果该box已经被抑制, 则跳过
                    if suppress[box]:
                        continue

                    # 抑制那些与本box重叠度大于阈值的box
                    # 找出这些box的索引, 并抑制
                    # suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # 旧版torch没有bool张量, 因此旧版采用上面的写法
                    suppress = suppress | (overlap[box] > max_overlap)

                    # 本box不需要被抑制
                    suppress[box] = False

                # 保存没被抑制的box
                # image_boxes.append(class_decoded_locs[1 - suppress])
                # image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                # image_scores.append(class_scores[1 - suppress])

                image_boxes.append(class_decoded_locs[~ suppress])
                image_labels.append(torch.LongTensor((~ suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~ suppress])

            # 如果在任何类别中都找不到目标, 则将其作为background
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size

class MultiBoxLoss(nn.Module):
    """
    多定位框损失

    这个类结合了:
    (1) 定位框的定位损失
    (2) 类别的置信度损失
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()  # 论文中使用, 平滑L1损失
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        前向传播.

        :param predicted_locs: 与8732个预定位框相关的预测定位框 (N, 8732, 4)
        :param predicted_scores: 每个定位框的类别概率 (N, 8732, n_classes)
        :param boxes: 正样本的定位框坐标,  N个tensors的list
        :param labels: 正样本标签, N个tensors的list
        :return: 多定位框损失, 标量
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # 遍历图片
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # 对于每一个预定位框, 找出与其重叠度最高的正样本
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # 我们不希望一个目标没有被非背景的预定位框所代表
            # 1.一个目标可能不是所有预定位框的最佳对象, 因此它不在object_for_each_prior中
            # 2.所有含有目标的预定位框可能根据阈值(0.5)被分配为背景

            # 为了解决这个问题-
            # 首先,找出每个对象的重叠度最高的预定位框
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
