import torch
from torch import nn
from torch.nn import Sequential
import functools
import time
from framework.utils import change_default_args


### FPN

class PointNet(nn.Module):
    def __init__(self, num_input_features, voxel_size, offset):
        super().__init__()
        self.name = 'PointNet'
        num_input_features += 5  # 9
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        # Create PillarFeatureNet layers
        in_channels = num_input_features
        self.out_channels = 64
        model = [nn.Conv1d(in_channels, self.out_channels, kernel_size=1, padding=0, bias=False),
                 nn.BatchNorm1d(self.out_channels),
                 nn.ReLU(True)]

        self.pfn_layers = nn.Sequential(*model)

    def forward(self, voxels, num_point_per_voxel, coors):
        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # coors [X Y Z Batch]
        f_center = torch.zeros_like(voxels[:, :, :2])
        f_center[:, :, 0] = voxels[:, :, 0] - (coors[:, 0].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (coors[:, 1].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [voxels, f_cluster, f_center]

        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        num_point_per_voxel = torch.unsqueeze(num_point_per_voxel, -1)
        max_point_per_voxel = features.shape[1]
        max_point_per_voxel = torch.arange(max_point_per_voxel, dtype=torch.int,
                                           device=num_point_per_voxel.device).view(1, -1)
        mask = num_point_per_voxel.int() > max_point_per_voxel
        # mask = get_paddings_indicator(num_point_per_voxel, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        x = features.permute(0, 2, 1).contiguous()
        x = self.pfn_layers(x).permute(0, 2, 1).contiguous()
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        return x_max.squeeze()


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 batch_size,
                 output_shape,
                 num_input_features=64):
        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.nx = output_shape[0]
        self.ny = output_shape[1]
        self.num_channels = num_input_features
        self.batch_size = batch_size

    def forward(self, voxel_features, coords):
        # batch_canvas will be the final output.
        batch_canvas = []

        for batch_itt in range(self.batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.num_channels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            if self.batch_size > 1:
                batch_mask = coords[:, 3] == batch_itt
                this_coords = coords[batch_mask, :]
                indices = this_coords[:, 1] * self.ny + this_coords[:, 2]
                indices = indices.type(torch.long)
                voxels = voxel_features[batch_mask, :]
            else:
                indices = coords[:, 0] * self.ny + coords[:, 1]
                indices = indices.type(torch.long)
                voxels = voxel_features

            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(self.batch_size, self.num_channels, self.nx, self.ny)

        return batch_canvas


class RPN(nn.Module):
    def __init__(self, num_rpn_input_filters):
        super().__init__()
        # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        layer_nums = [2, 4, 4]
        layer_strides = [2, 2, 2]
        num_filters = [64, 128, 256]
        upsample_strides = [1, 2, 4]
        num_upsample_filters = [64, 128, 128]
        num_input_filters = num_rpn_input_filters
        use_direction_classifier = True
        self._use_direction_classifier = use_direction_classifier
        self.out_plane = sum(num_upsample_filters)

        norm_layer = change_default_args(
            eps=1e-3, momentum=0.01)(nn.InstanceNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)
        ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)

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


class FPN(nn.Module):
    def __init__(self, num_rpn_input_filters):
        super().__init__()
        # norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        layer_nums = [2, 4, 4]
        layer_strides = [2, 2, 2]
        num_filters = [64, 128, 256]
        upsample_strides = [1, 2, 4]
        num_upsample_filters = [64, 128, 128]
        num_input_filters = num_rpn_input_filters
        use_direction_classifier = True
        self._use_direction_classifier = use_direction_classifier
        self.out_plane = 256

        norm_layer = change_default_args(eps=1e-3, momentum=0.01)(nn.InstanceNorm2d)
        Conv2d = change_default_args(bias=False)(nn.Conv2d)
        ConvTranspose2d = change_default_args(bias=False)(nn.ConvTranspose2d)

        model = [Conv2d(num_input_filters, num_filters[0], 3, stride=2, padding=1),
                 norm_layer(num_filters[0]),
                 nn.ReLU()]
        model += [Resnet2(num_filters[0], norm_layer, 1)]
        model += [Resnet2(num_filters[0], norm_layer, 0)]
        self.block1 = Sequential(*model)

        model = [Conv2d(num_filters[0], self.out_plane, 1),
                 norm_layer(num_filters[0]),
                 nn.ReLU()]
        self.block11 = Sequential(*model)

        model = [Conv2d(num_filters[0], num_filters[1], 3, stride=layer_strides[1], padding=1),
                 norm_layer(num_filters[1]),
                 nn.ReLU()]
        model += [Resnet2(num_filters[1], norm_layer, 1)]
        model += [Resnet2(num_filters[1], norm_layer, 1)]
        model += [Resnet2(num_filters[1], norm_layer, 0)]
        self.block2 = Sequential(*model)

        model = [Conv2d(num_filters[1], self.out_plane, 1),
                 norm_layer(num_filters[1]),
                 nn.ReLU()]
        self.block21 = Sequential(*model)

        model = [ConvTranspose2d(num_filters[1], self.out_plane, 2, stride=2),
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

        model = [Conv2d(num_filters[2], self.out_plane, 1),
                 norm_layer(num_filters[2]),
                 nn.ReLU()]
        self.block31 = Sequential(*model)

        model = [ConvTranspose2d(self.out_plane, self.out_plane, 2, stride=2),
                 norm_layer(num_upsample_filters[2]),
                 nn.ReLU()]
        self.deconv3 = Sequential(*model)

    def forward(self, x):
        x = self.block1(x)
        l1 = self.block11(x)
        x = self.block2(x)
        l2 = self.block21(x)
        x = self.block3(x)
        r3 = self.block31(x)

        r2 = self.deconv3(r3) + l2
        r1 = self.deconv2(r2) + l1
        #out = [r3, r2, r1]
        return r1


class SingleHead(nn.Module):

    def __init__(self, in_plane):
        super().__init__()
        self.box_code_size = 7

        num_ped_size = 1
        num_ped_rot = 1
        num_ped_anchor_per_loc = num_ped_size * num_ped_rot
        self.conv_ped_cls = nn.Conv2d(in_plane, num_ped_anchor_per_loc, 1)
        self.conv_ped_box = nn.Conv2d(in_plane, num_ped_anchor_per_loc * self.box_code_size, 1)
        self.conv_ped_dir = nn.Conv2d(in_plane, num_ped_anchor_per_loc * 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        ped_cls_preds = self.conv_ped_cls(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        ped_box_preds = self.conv_ped_box(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_code_size)
        ped_dir_preds = self.conv_ped_dir(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        cls_preds = ped_cls_preds
        box_preds = ped_box_preds
        dir_cls_preds = ped_dir_preds

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_cls_preds": dir_cls_preds
        }

        return pred_dict


class SingleHeads(nn.Module):

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

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return pred_dict


# class SingleHeads(nn.Module):
#
#     def __init__(self, in_plane):
#         super().__init__()
#         self.box_code_size = 7
#
#         num_veh_size = 3
#         num_veh_rot = 2
#         self.num_veh_anchor_per_loc = num_veh_size * num_veh_rot
#
#         num_ped_size = 1
#         num_ped_rot = 1
#         self.num_ped_anchor_per_loc = num_ped_size * num_ped_rot
#
#         num_cyc_size = 1
#         num_cyc_rot = 2
#         self.num_cyc_anchor_per_loc = num_cyc_size * num_cyc_rot
#
#         num_anchor_per_loc = self.num_veh_anchor_per_loc + self.num_ped_anchor_per_loc + self.num_cyc_anchor_per_loc
#
#         self.conv_cls = nn.Conv2d(in_plane, num_anchor_per_loc, 1)
#         self.conv_box = nn.Conv2d(in_plane, num_anchor_per_loc * self.box_code_size, 1)
#         self.conv_dir = nn.Conv2d(in_plane, num_anchor_per_loc * 2, 1)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#
#         cls_preds = self.conv_cls(x)
#         start = 0
#         end = self.num_veh_anchor_per_loc
#         veh_cls_preds = cls_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
#         start = end
#         end = start + self.num_ped_anchor_per_loc
#         ped_cls_preds = cls_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
#         start = end
#         end = start + self.num_cyc_anchor_per_loc
#         cyc_cls_preds = cls_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
#
#         box_preds = self.conv_box(x)
#         start = 0
#         end = self.num_veh_anchor_per_loc * self.box_code_size
#         veh_box_preds = box_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
#                                                                                            self.box_code_size)
#         start = end
#         end = start + self.num_ped_anchor_per_loc * self.box_code_size
#         ped_box_preds = box_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
#                                                                                            self.box_code_size)
#         start = end
#         end = start + self.num_cyc_anchor_per_loc * self.box_code_size
#         cyc_box_preds = box_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
#                                                                                            self.box_code_size)
#
#         dir_preds = self.conv_dir(x)
#         start = 0
#         end = self.num_veh_anchor_per_loc * 2
#         veh_dir_preds = dir_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
#         start = end
#         end = start + self.num_ped_anchor_per_loc * 2
#         ped_dir_preds = dir_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
#         start = end
#         end = start + self.num_cyc_anchor_per_loc * 2
#         cyc_dir_preds = dir_preds[:, start:end, ...].permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
#
#         cls_preds = torch.cat((veh_cls_preds, ped_cls_preds, cyc_cls_preds), dim=1)
#         box_preds = torch.cat((veh_box_preds, ped_box_preds, cyc_box_preds), dim=1)
#         dir_preds = torch.cat((veh_dir_preds, ped_dir_preds, cyc_dir_preds), dim=1)
#
#         pred_dict = {
#             "cls_preds": cls_preds,
#             "box_preds": box_preds,
#             "dir_preds": dir_preds
#         }
#
#         return pred_dict


class MultiHead(nn.Module):

    def __init__(self, in_plane):
        super().__init__()
        self.box_code_size = 7

        num_veh_size = 3
        num_veh_rot = 2
        num_veh_anchor_per_loc = num_veh_size * num_veh_rot
        self.conv_veh_cls = nn.Conv2d(in_plane, num_veh_anchor_per_loc, 1)
        self.conv_veh_box = nn.Conv2d(in_plane, num_veh_anchor_per_loc * self.box_code_size, 1)
        self.conv_veh_dir = nn.Conv2d(in_plane, num_veh_anchor_per_loc * 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_preds = self.conv_veh_cls(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        box_preds = self.conv_veh_box(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_code_size)
        dir_preds = self.conv_veh_dir(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return pred_dict


class MultiHeads(nn.Module):

    def __init__(self, in_plane):
        super().__init__()
        self.box_code_size = 7

        num_veh_size = 3
        num_veh_rot = 2
        num_veh_anchor_per_loc = num_veh_size * num_veh_rot
        self.conv_veh_cls = nn.Conv2d(in_plane, num_veh_anchor_per_loc, 1)
        self.conv_veh_box = nn.Conv2d(in_plane, num_veh_anchor_per_loc * self.box_code_size, 1)
        self.conv_veh_dir = nn.Conv2d(in_plane, num_veh_anchor_per_loc * 2, 1)

        num_ped_size = 1
        num_ped_rot = 1
        num_ped_anchor_per_loc = num_ped_size * num_ped_rot
        self.conv_ped_cls = nn.Conv2d(in_plane, num_ped_anchor_per_loc, 1)
        self.conv_ped_box = nn.Conv2d(in_plane, num_ped_anchor_per_loc * self.box_code_size, 1)
        self.conv_ped_dir = nn.Conv2d(in_plane, num_ped_anchor_per_loc * 2, 1)

        num_cyc_size = 1
        num_cyc_rot = 2
        num_cyc_anchor_per_loc = num_cyc_size * num_cyc_rot
        self.conv_cyc_cls = nn.Conv2d(in_plane, num_cyc_anchor_per_loc, 1)
        self.conv_cyc_box = nn.Conv2d(in_plane, num_cyc_anchor_per_loc * self.box_code_size, 1)
        self.conv_cyc_dir = nn.Conv2d(in_plane, num_cyc_anchor_per_loc * 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        veh_cls_preds = self.conv_veh_cls(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        veh_box_preds = self.conv_veh_box(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_code_size)
        veh_dir_preds = self.conv_veh_dir(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        ped_cls_preds = self.conv_ped_cls(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        ped_box_preds = self.conv_ped_box(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_code_size)
        ped_dir_preds = self.conv_ped_dir(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        cyc_cls_preds = self.conv_cyc_cls(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        cyc_box_preds = self.conv_cyc_box(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_code_size)
        cyc_dir_preds = self.conv_cyc_dir(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        cls_preds = torch.cat((veh_cls_preds, ped_cls_preds, cyc_cls_preds), dim=1)
        box_preds = torch.cat((veh_box_preds, ped_box_preds, cyc_box_preds), dim=1)
        dir_preds = torch.cat((veh_dir_preds, ped_dir_preds, cyc_dir_preds), dim=1)

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return pred_dict


class DiffHeads(nn.Module):

    def __init__(self, in_plane):
        super().__init__()
        self.box_code_size = 7

        self.feature1_anchor_per_loc = 2 * 2
        self.conv_feature1_cls = nn.Conv2d(in_plane, self.feature1_anchor_per_loc, 3, stride=2, padding=1)
        self.conv_feature1_box = nn.Conv2d(in_plane, self.feature1_anchor_per_loc * 7, 3, stride=2, padding=1)
        self.conv_feature1_dir = nn.Conv2d(in_plane, self.feature1_anchor_per_loc * 2, 3, stride=2, padding=1)

        # self.feature1_anchor_per_loc = 2 * 2
        # self.conv_feature1_cls = nn.Conv2d(in_plane, self.feature1_anchor_per_loc, 1)
        # self.conv_feature1_box = nn.Conv2d(in_plane, self.feature1_anchor_per_loc * 7, 1)
        # self.conv_feature1_dir = nn.Conv2d(in_plane, self.feature1_anchor_per_loc * 2, 1)

        self.feature2_anchor_per_loc = 1 * 2 + 1 * 1 + 1 * 2
        self.conv_feature2_cls = nn.Conv2d(in_plane, self.feature2_anchor_per_loc, 1)
        self.conv_feature2_box = nn.Conv2d(in_plane, self.feature2_anchor_per_loc * 7, 1)
        self.conv_feature2_dir = nn.Conv2d(in_plane, self.feature2_anchor_per_loc * 2, 1)

    def forward(self, x):
        N = x.shape[0]
        cls_preds1 = self.conv_feature1_cls(x).view(N, -1, 1)

        box_preds1 = self.conv_feature1_box(x)
        N, C, H, W = box_preds1.shape
        box_preds1 = box_preds1.view(N, self.feature1_anchor_per_loc, self.box_code_size, H, W).permute(0, 1, 3, 4, 2)
        box_preds1 = box_preds1.contiguous().view(N, -1, self.box_code_size)

        dir_preds1 = self.conv_feature1_dir(x)
        N, C, H, W = dir_preds1.shape
        dir_preds1 = dir_preds1.view(N, self.feature1_anchor_per_loc, 2, H, W).permute(0, 1, 3, 4, 2)
        dir_preds1 = dir_preds1.contiguous().view(N, -1, 2)

        cls_preds2 = self.conv_feature2_cls(x).view(N, -1, 1)

        box_preds2 = self.conv_feature2_box(x)
        N, C, H, W = box_preds2.shape
        box_preds2 = box_preds2.view(N, self.feature2_anchor_per_loc, self.box_code_size, H, W).permute(0, 1, 3, 4, 2)
        box_preds2 = box_preds2.contiguous().view(N, -1, self.box_code_size)

        dir_preds2 = self.conv_feature2_dir(x)
        N, C, H, W = dir_preds2.shape
        dir_preds2 = dir_preds2.view(N, self.feature2_anchor_per_loc, 2, H, W).permute(0, 1, 3, 4, 2)
        dir_preds2 = dir_preds2.contiguous().view(N, -1, 2)

        cls_preds = torch.cat((cls_preds1, cls_preds2), dim=1)
        box_preds = torch.cat((box_preds1, box_preds2), dim=1)
        dir_preds = torch.cat((dir_preds1, dir_preds2), dim=1)

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return pred_dict


class FPNHeads(nn.Module):

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

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return pred_dict

class PointPillars(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        self.pillar_point_net = PointNet(config['num_point_features'], config['voxel_size'], config['detection_offset'])
        num_rpn_input_filters = self.pillar_point_net.out_channels
        self.middle_feature_extractor = PointPillarsScatter(batch_size=config['batch_size'],
                                                            output_shape=config['grid_size'],
                                                            num_input_features=num_rpn_input_filters)

        self.rpn = FPN(num_rpn_input_filters)
        # self.heads = SingleHeads(self.rpn.out_plane)
        self.heads = FPNHeads(self.rpn.out_plane)
        self.voxel_features_time = 0.0
        self.spatial_features_time = 0.0
        self.rpn_feature_time = 0.0
        self.heads_time = 0.0

    def forward(self, example):
        start = time.time()
        voxel_features = self.pillar_point_net(example["voxels"], example["num_points_per_voxel"],
                                               example["coordinates"])
        torch.cuda.synchronize()
        voxel_features_time = time.time()
        spatial_features = self.middle_feature_extractor(voxel_features, example["coordinates"])
        torch.cuda.synchronize()
        spatial_features_time = time.time()
        rpn_feature = self.rpn(spatial_features)
        torch.cuda.synchronize()
        rpn_feature_time = time.time()
        preds_dict = self.heads(rpn_feature)
        torch.cuda.synchronize()
        heads_time = time.time()

        self.voxel_features_time += voxel_features_time - start
        self.spatial_features_time += spatial_features_time - voxel_features_time
        self.rpn_feature_time += rpn_feature_time - spatial_features_time
        self.heads_time += heads_time - rpn_feature_time

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
