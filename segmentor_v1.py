import functools
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm22', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        #print(new_features.shape)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            # print('--------------------------------------------------%d' %i)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool_norm', nn.BatchNorm3d(num_output_features))
        self.add_module('pool_relu', nn.ReLU(inplace=True))
        self.add_module('pool', nn.Conv3d(num_output_features, num_output_features,
                                          kernel_size=2, stride=2))

class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.channel_excitation = nn.Sequential(nn.Conv3d(channel, channel // reduction, kernel_size=1, padding=0),
                                                # nn.BatchNorm3d(channel//reduction),
                                                nn.ReLU(inplace=True),
                                                nn.Conv3d(channel // reduction, channel, kernel_size=1, padding=0),
                                                nn.Softmax(dim=1))

        self.spatial_se = nn.Sequential(nn.Conv3d(channel, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Softmax(dim=1))

    def forward(self, x):
        chn_se = self.avg_pool(x)
        chn_se = self.channel_excitation(chn_se)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)

        out_se = chn_se + spa_se
        # print (out_se.shape)
        return out_se


class DenseNet(nn.Module):

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=3):
        super(DenseNet, self).__init__()
        # First three convolutions
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(2, num_init_features, kernel_size=3, stride=1, padding=1)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),

            ('conv1', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm1', nn.BatchNorm3d(num_init_features)),
            ('relu1', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv3d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm2', nn.BatchNorm3d(num_init_features)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.conv_pool_first = nn.Conv3d(num_init_features, num_init_features,
                                         kernel_size=2, stride=2, padding=0,
                                         bias=False)
        self.conv_pool_first = nn.MaxPool3d(kernel_size=2, stride=2)

        # Each denseblock
        num_features = num_init_features
        num_features_list = []
        self.dense_blocks = nn.ModuleList([])
        self.scse_blocks = nn.ModuleList([])
        self.transit_blocks = nn.ModuleList([])
        self.upsampling_blocks = nn.ModuleList([])
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.dense_blocks.append(block);  # print(self.dense_blocks[1], '---------------------0000')
            num_features = num_features + num_layers * growth_rate
            # se = SEModule(num_features, reduction=8)
            scse = SCSEBlock(num_features, reduction=16)
            self.scse_blocks.append(scse)

            up_block = nn.ConvTranspose3d(num_features, num_classes,
                                          kernel_size=2 ** (i + 1) + 2,
                                          stride=2 ** (i + 1),
                                          padding=1, groups=1, bias=False)
            self.upsampling_blocks.append(up_block)
            num_features_list.append(num_features)

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.transit_blocks.append(trans)
                num_features = num_features // 2

        # ----------------------- classifier -----------------------
        self.bn_class_3_3_first = nn.BatchNorm3d(num_init_features)
        self.conv_class_3_3_first = nn.Conv3d(num_init_features, num_classes, kernel_size=3, padding=1)

        self.bn_class_3_3 = nn.BatchNorm3d(num_classes * 5)
        self.conv_class_3_3 = nn.Conv3d(num_classes * 5, num_classes * 5, kernel_size=3, padding=1)

        self.bn_class = nn.BatchNorm3d(num_classes * 5)
        self.conv_class = nn.Conv3d(num_classes * 5, num_classes, kernel_size=1, padding=0)

        if self.training:
            self.bn_class_aux1 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux1 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

            self.bn_class_aux2 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux2 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

            self.bn_class_aux3 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux3 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

            self.bn_class_aux4 = nn.BatchNorm3d(num_classes)
            self.conv_class_aux4 = nn.Conv3d(num_classes, num_classes, kernel_size=1, padding=0)

        # ----------------------------------------------------------
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # print(self.training)

    def forward(self, x):
        x1 = self.features(x);
        x1_cross = self.conv_pool_first(x1);
        # out = self.scse_first(out)

        #        Block_1
        out = self.dense_blocks[0](x1)
        out = self.scse_blocks[0](out)
        up_block1 = self.upsampling_blocks[0](out)  # no used
        x2 = self.transit_blocks[0](out)
        x2_cross = self.conv_pool_first(x2)

        #        Block_2
        out = self.dense_blocks[1](x2)
        out = self.scse_blocks[1](out)
        up_block2 = self.upsampling_blocks[1](out)
        x3 = self.transit_blocks[1](out)

        #        Block_3
        out = self.dense_blocks[2](x3)
        out = self.scse_blocks[2](out)
        up_block3 = self.upsampling_blocks[2](out)
        x4 = self.transit_blocks[2](out)

        return x4, x3, x2_cross, x2, x1_cross, x1

class expansive(nn.Module):

    def __init__(self, in_channel_block41=120, out_channel_block41=128, out_channel_block20=128,
                 out_channel_block10=128, out_channel_block0=32, norm_layer=nn.BatchNorm3d,
                 out_channel_block40=256, num_classes=4):
        super(expansive, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.block_up_41 = nn.Sequential(OrderedDict([
            # ('relu41', nn.ReLU(True),
            ('conv41', nn.ConvTranspose3d(in_channel_block41, out_channel_block41, kernel_size=4, stride=2, padding=1,
                                          bias=False)),
            ('norm41', norm_layer(out_channel_block41)),
        ]))
        in_channel_block41_more = 464  # 384
        out_channel_block41_more = out_channel_block20 * 2
        self.block_up_41_more = nn.Sequential(
            OrderedDict([
                ('relu41', nn.ReLU(True)),
                ('conv41', nn.Conv3d(in_channel_block41_more, out_channel_block41_more, kernel_size=3, stride=1,
                                     padding=1, bias=False)),
                ('norm41', norm_layer(out_channel_block41_more)),
        ]))

        in_channel_block21 = out_channel_block41_more
        out_channel_block21 = in_channel_block21
        self.block_up_21 = nn.Sequential(OrderedDict([
            ('relu21', nn.ReLU(True)),
            ('conv21', nn.ConvTranspose3d(in_channel_block21, out_channel_block21, kernel_size=4, stride=2, padding=1,
                                          bias=False)),
            ('norm21', norm_layer(out_channel_block21))
        ]))
        in_channel_block21_more = 320
        out_channel_block21_more = 128
        self.block_up_21_more = nn.Sequential(
            OrderedDict([
                ('relu21', nn.ReLU(True)),
                ('conv21', nn.Conv3d(in_channel_block21_more, out_channel_block21_more, kernel_size=3, stride=1,
                                     padding=1, bias=False)),
                ('norm21', norm_layer(out_channel_block21_more)),
        ]))

        in_channel_block11 = 128
        out_channel_block11 = in_channel_block11 // 2
        self.block_up_11 = nn.Sequential(OrderedDict([
            ('relu11', nn.ReLU(True)),
            ('conv11', nn.ConvTranspose3d(in_channel_block11, out_channel_block11, kernel_size=4, stride=2, padding=1,
                                          bias=False)),
            ('norm11', norm_layer(out_channel_block11))
        ]))
        in_channel_block11_more = in_channel_block11
        out_channel_block11_more = 128
        self.block_up_11_more = nn.Sequential(
            OrderedDict([
                ('relu41', nn.ReLU(True)),
                ('conv41', nn.Conv3d(in_channel_block11_more, out_channel_block11_more, kernel_size=3, stride=1,
                                     padding=1, bias=False)),
                ('norm41', norm_layer(out_channel_block11_more)),
        ]))


        in_channel_classifier = out_channel_block11 + out_channel_block0 * 2
        self.classifier = nn.Sequential(OrderedDict([
            ('relu00', nn.ReLU(True)),
            ('conv00', nn.Conv3d(in_channel_classifier, num_classes, kernel_size=1, stride=1, padding=0, bias=False)),

        ]))

    def forward(self, x4, x3, x2_cross, x2, x1_cross, x1):

        x = (self.block_up_41(x4))

        x3 = torch.cat([x, x3], dim=1);
        x3_cross = torch.cat([x, x2_cross], dim=1)
        x3 = torch.cat([x3, x3_cross], dim=1)
        x3 = self.block_up_41_more(x3)

        x = self.block_up_21(x3);
        x2 = torch.cat([x, x2], dim=1);  # no used
        x2_output = torch.cat([x, x1_cross], 1)
        x2 = self.block_up_21_more(x2_output);

        x = self.block_up_11(x2);
        x1 = torch.cat([x, x1], dim=1)
        x1 = self.block_up_11_more(x1)

        x = self.classifier(x1);
        return x


class Unet_dense(nn.Module):
    def __init__(self):
        super(Unet_dense, self).__init__()
        self.den = DenseNet()
        self.expanding = expansive()

    def forward(self, x):
        x4, x3,x2_cross, x2,x1_cross, x1 = self.den(x)
        x = self.expanding(x4, x3,x2_cross, x2,x1_cross, x1)
        #print(x.shape, '---------------trung-------------')
        return x




