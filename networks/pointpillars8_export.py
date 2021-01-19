import torch
from torch import nn
from torch.nn import Sequential
import functools
import time
from framework.utils import change_default_args
from framework.trt_utils import export_onnx, build_engine, load_engine


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Create PillarFeatureNet layers
        in_channels = 9
        self.out_channels = 64
        model = [nn.Conv1d(in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
                 nn.BatchNorm1d(self.out_channels),
                 nn.ReLU(True)]

        self.pfn_layers = nn.Sequential(*model)

    def forward(self, features):
        # Forward pass through PFNLayers
        x = features.permute(0, 2, 1).contiguous()
        x = self.pfn_layers(x)

        x = x.permute(2, 1, 0).contiguous()
        x_max = torch.max(x, dim=0, keepdim=True)[0]
        # x_max = x_max.squeeze()
        return x_max


class Scatter:
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        super().__init__()
        self.output_shape = output_shape
        self.nx = output_shape[0]
        self.ny = output_shape[1]
        self.num_channels = num_input_features

    def run(self, voxel_features, coords):
        canvas = torch.zeros(self.num_channels, self.nx * self.ny, dtype=voxel_features.dtype,
                             device=voxel_features.device)
        indices = coords[:, 0] * self.ny + coords[:, 1]
        indices = indices.type(torch.long)
        canvas[:, indices] = voxel_features
        canvas = canvas.contiguous().view(1, self.num_channels, self.nx, self.ny)
        return canvas


class RPN(nn.Module):
    def __init__(self, num_rpn_input_filters):
        super().__init__()

        layer_strides = [2, 2, 2]
        num_filters = [64, 128, 256]
        upsample_strides = [1, 2, 4]
        num_upsample_filters = [64, 128, 128]  # [128, 128, 128]
        num_input_filters = num_rpn_input_filters

        self.out_plane = sum(num_upsample_filters)
        norm_layer = functools.partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        # norm_layer = functools.partial(nn.InstanceNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = functools.partial(nn.Conv2d, bias=False)
        ConvTranspose2d = functools.partial(nn.ConvTranspose2d, bias=False)

        model = [Conv2d(num_input_filters, num_filters[0], 3, stride=2, padding=1),
                 norm_layer(num_filters[0]),
                 nn.ReLU()]
        model += [Resnet2(num_filters[0], norm_layer, 1)]
        model += [Resnet2(num_filters[0], norm_layer, 0)]
        self.block1 = Sequential(*model)

        model = [ConvTranspose2d(num_filters[0], num_upsample_filters[0], upsample_strides[0],
                                 stride=upsample_strides[0]),
                 norm_layer(num_upsample_filters[0]),
                 nn.ReLU()]
        self.deconv1 = Sequential(*model)

        model = [Conv2d(num_filters[0], num_filters[1], 3, stride=layer_strides[1], padding=1),
                 norm_layer(num_filters[1]),
                 nn.ReLU()]
        model += [Resnet2(num_filters[1], norm_layer, 1)]
        model += [Resnet2(num_filters[1], norm_layer, 1)]
        model += [Resnet2(num_filters[1], norm_layer, 0)]
        self.block2 = Sequential(*model)

        model = [ConvTranspose2d(num_filters[1], num_upsample_filters[1], upsample_strides[1],
                                 stride=upsample_strides[1]),
                 norm_layer(num_upsample_filters[1]),
                 nn.ReLU()]
        self.deconv2 = Sequential(*model)

        model = [Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2], padding=1),
                 norm_layer(num_filters[2]),
                 nn.ReLU()]
        model += [Resnet2(num_filters[2], norm_layer, 1)]
        model += [Resnet2(num_filters[2], norm_layer, 1)]
        model += [Resnet2(num_filters[2], norm_layer, 0)]
        self.block3 = Sequential(*model)

        model = [ConvTranspose2d(num_filters[2], num_upsample_filters[2], upsample_strides[2],
                                 stride=upsample_strides[2]),
                 norm_layer(num_upsample_filters[2]),
                 nn.ReLU()]
        self.deconv3 = Sequential(*model)

    def forward(self, x):
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        return x


class SharedHead(nn.Module):

    def __init__(self, in_plane):
        super().__init__()
        self.box_code_size = 7

        num_veh_size = 3
        num_veh_rot = 2
        self.num_veh_anchor_per_loc = num_veh_size * num_veh_rot

        num_ped_size = 1
        num_ped_rot = 1
        self.num_ped_anchor_per_loc = num_ped_size * num_ped_rot

        num_cyc_size = 1
        num_cyc_rot = 2
        self.num_cyc_anchor_per_loc = num_cyc_size * num_cyc_rot

        self.num_anchor_per_loc = self.num_veh_anchor_per_loc + self.num_ped_anchor_per_loc + self.num_cyc_anchor_per_loc

        self.conv_cls = nn.Conv2d(in_plane, self.num_anchor_per_loc, 1)
        self.conv_box = nn.Conv2d(in_plane, self.num_anchor_per_loc * self.box_code_size, 1)
        self.conv_dir = nn.Conv2d(in_plane, self.num_anchor_per_loc * 2, 1)

    def forward(self, x):
        N = x.shape[0]

        cls_preds = self.conv_cls(x).view(N, -1, 1)

        box_preds = self.conv_box(x)
        N, C, H, W = box_preds.shape
        box_preds = box_preds.view(N, self.num_anchor_per_loc, self.box_code_size, H, W).permute(0, 1, 3, 4, 2)
        box_preds = box_preds.contiguous().view(N, -1, self.box_code_size)

        dir_preds = self.conv_dir(x)
        N, C, H, W = dir_preds.shape
        dir_preds = dir_preds.view(N, self.num_anchor_per_loc, 2, H, W).permute(0, 1, 3, 4, 2).contiguous().view(N, -1,
                                                                                                                 2)
        return cls_preds, box_preds, dir_preds


class PointPillars(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        voxel_size = config['voxel_size']
        offset = config['detection_offset']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        self.pillar_point_net = PointNet()
        self.scatter = Scatter(output_shape=config['grid_size'], num_input_features=self.pillar_point_net.out_channels)
        self.rpn = RPN(self.pillar_point_net.out_channels)
        self.heads = SharedHead(self.rpn.out_plane)
        self.pfn_time, self.rpn_time, self.scatter_time, self.heads_time = 0.0, 0.0, 0.0, 0.0

    def forward(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]
        # prepare pillars feature
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxels[:, :, :2])
        f_center[:, :, 0] = voxels[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (coors[:, 1].float().unsqueeze(1) * self.vy + self.y_offset)
        features_ls = [voxels, f_cluster, f_center]
        features = torch.cat(features_ls, dim=-1)
        num_point_per_voxel = torch.unsqueeze(num_point_per_voxel, -1)
        max_point_per_voxel = features.shape[1]
        max_point_per_voxel = torch.arange(max_point_per_voxel, dtype=torch.int,
                                           device=num_point_per_voxel.device).view(1, -1)
        mask = num_point_per_voxel.int() > max_point_per_voxel
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        start = time.time()
        pfn_out = self.pillar_point_net(features)
        torch.cuda.synchronize()
        pfn_time = time.time()
        voxel_feature = self.scatter.run(pfn_out, coors)
        torch.cuda.synchronize()
        scatter_time = time.time()
        rpn_out = self.rpn(voxel_feature)
        torch.cuda.synchronize()
        rpn_time = time.time()
        cls_preds, box_preds, dir_preds = self.heads(rpn_out)
        torch.cuda.synchronize()
        heads_time = time.time()

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        self.pfn_time += pfn_time - start
        self.scatter_time += scatter_time - pfn_time
        self.rpn_time += rpn_time - scatter_time
        self.heads_time += heads_time - rpn_time

        return preds_dict

    def export(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]
        # prepare pillars feature
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxels[:, :, :2])
        f_center[:, :, 0] = voxels[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (coors[:, 1].float().unsqueeze(1) * self.vy + self.y_offset)
        features_ls = [voxels, f_cluster, f_center]
        features = torch.cat(features_ls, dim=-1)
        num_point_per_voxel = torch.unsqueeze(num_point_per_voxel, -1)
        max_point_per_voxel = features.shape[1]
        max_point_per_voxel = torch.arange(max_point_per_voxel, dtype=torch.int,
                                           device=num_point_per_voxel.device).view(1, -1)
        mask = num_point_per_voxel.int() > max_point_per_voxel
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        pfn_out = self.pillar_point_net(features).squeeze(dim=0)
        pfn_onnx_file_path = '../deployment/pfn_test.onnx'
        pfn_engine_path = '../deployment/pfn16_test.engine'
        export_onnx(self.pillar_point_net, features, pfn_onnx_file_path, dynamic=True)
        build_engine(pfn_onnx_file_path, pfn_engine_path, dynamic=True)
        # exit(0)

        voxel_feature = self.scatter.run(pfn_out, coors)

        rpn_feature = self.rpn(voxel_feature)
        rpn_onnx_file_path = '../deployment/rpn_test.onnx'
        rpn_engine_path = '../deployment/rpn16_test.engine'
        export_onnx(self.rpn, voxel_feature, rpn_onnx_file_path, dynamic=False)
        build_engine(rpn_onnx_file_path, rpn_engine_path, dynamic=False)
        # exit(0)

        cls_preds, box_preds, dir_preds = self.heads(rpn_feature)
        head_onnx_file_path = '../deployment/head_test.onnx'
        head_engine_path = '../deployment/head16_test.engine'
        export_onnx(self.heads, rpn_feature, head_onnx_file_path, dynamic=False, input_names=['inputs'],
                    output_names=['cls_preds', 'box_preds', 'dir_preds'])
        build_engine(head_onnx_file_path, head_engine_path, dynamic=False)
        exit(0)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return preds_dict


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Resnet(nn.Module):
    def __init__(self, dim, norm_layer):
        super(Resnet, self).__init__()
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1), norm_layer(dim)]
        self.conv_block = nn.Sequential(*conv_block)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.conv_block(x)
        out = self.relu(x)
        return out


class Resnet2(nn.Module):

    def __init__(self, dim, norm_layer, num_layer=1):
        ### Full pre-activation
        super(Resnet2, self).__init__()
        Conv2d = change_default_args(bias=False)(nn.Conv2d)
        conv_block = [norm_layer(dim), nn.ReLU(True), Conv2d(dim, dim, kernel_size=3, padding=1)]
        for layer in range(num_layer):
            conv_block += [norm_layer(dim), nn.ReLU(True), Conv2d(dim, dim, kernel_size=3, padding=1)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
