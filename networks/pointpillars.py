
import torch
from torch import nn
from torch.nn import Sequential
import functools


class PointNet(nn.Module):
    def __init__(self, num_input_features, voxel_size, offset):
        super().__init__()
        self.name = 'PointNet'

        num_input_features += 5  # 9
        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        # try to put in dataloader use numba
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        # Create PillarFeatureNet layers
        in_channels = num_input_features
        out_channels = 64
        model = [nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                 nn.BatchNorm1d(out_channels),
                 nn.ReLU(True)]

        self.pfn_layers = nn.Sequential(*model)
        '''
        model = [nn.Conv1d(64, out_channels, kernel_size=1, padding=0, bias=False),
                 nn.BatchNorm1d(out_channels),
                 nn.ReLU(True)]

        self.pfn_layers2 = nn.Sequential(*model)
        '''
        self.out_channels = out_channels

    def forward(self, voxels, num_point_per_voxel, coors):
        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(voxels[:, :, :2])
        f_center[:, :, 0] = voxels[:, :, 0] - (coors[:, 1].float().unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = voxels[:, :, 1] - (coors[:, 2].float().unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [voxels, f_cluster, f_center]

        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        num_point_per_voxel = torch.unsqueeze(num_point_per_voxel, -1)
        max_point_per_voxel = features.shape[1]
        max_point_per_voxel = torch.arange(max_point_per_voxel, dtype=torch.int, device=num_point_per_voxel.device).view(1, -1)
        mask = num_point_per_voxel.int() > max_point_per_voxel
        # mask = get_paddings_indicator(num_point_per_voxel, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        x = features.permute(0, 2, 1).contiguous()
        x = self.pfn_layers(x).permute(0, 2, 1).contiguous()
        #x = self.pfn_layers2(x)
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

            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.ny + this_coords[:, 2]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
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
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        layer_nums = [3, 5, 5]
        layer_strides = [2, 2, 2]
        num_filters = [64, 128, 256]
        upsample_strides = [1, 2, 4]
        num_upsample_filters = [64, 128, 128]
        num_input_filters = num_rpn_input_filters
        use_direction_classifier = True
        self._use_direction_classifier = use_direction_classifier

        model = [nn.ZeroPad2d(1),
                 nn.Conv2d(num_input_filters, num_filters[0], 3, stride=2),
                 norm_layer(num_filters[0]),
                 nn.ReLU()]
        for i in range(layer_nums[0]):
            model += [nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1),
                      norm_layer(num_filters[0]),
                      nn.ReLU()]
        self.block1 = Sequential(*model)

        model = [nn.ConvTranspose2d(num_filters[0], num_upsample_filters[0], upsample_strides[0], stride=upsample_strides[0]),
                 norm_layer(num_upsample_filters[0]),
                 nn.ReLU()]
        self.deconv1 = Sequential(*model)

        model = [nn.ZeroPad2d(1),
                 nn.Conv2d(num_filters[0], num_filters[1], 3, stride=layer_strides[1]),
                 norm_layer(num_filters[1]),
                 nn.ReLU()]
        for i in range(layer_nums[1]):
            model += [nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1),
                      norm_layer(num_filters[1]),
                      nn.ReLU()]
            self.block2 = Sequential(*model)

        model = [nn.ConvTranspose2d(num_filters[1], num_upsample_filters[1], upsample_strides[1],
                               stride=upsample_strides[1]),
                 norm_layer(num_upsample_filters[1]),
                 nn.ReLU()]
        self.deconv2 = Sequential(*model)

        model = [nn.ZeroPad2d(1),
                 nn.Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
                 norm_layer(num_filters[2]),
                 nn.ReLU()]
        for i in range(layer_nums[2]):
            model += [nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1),
                      norm_layer(num_filters[2]),
                      nn.ReLU()]
        self.block3 = Sequential(*model)

        model = [nn.ConvTranspose2d(num_filters[2], num_upsample_filters[2], upsample_strides[2],
                               stride=upsample_strides[2]),
                 norm_layer(num_upsample_filters[2]),
                 nn.ReLU()]
        self.deconv3 = Sequential(*model)

        num_anchor_per_loc = 2
        num_class = 1
        num_cls = num_anchor_per_loc * num_class
        box_code_size = 7
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x):
        # x = self.block0(x)
        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        pred_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }

        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            pred_dict["dir_cls_preds"] = dir_cls_preds

        return pred_dict


class PointPillars(nn.Module):

    def __init__(self, config, loss_generator=None):
        super().__init__()
        self.device = config['device']
        self.pillar_point_net = PointNet(config['num_point_features'], config['voxel_size'], config['detection_offset'])
        num_rpn_input_filters = self.pillar_point_net.out_channels
        self.middle_feature_extractor = PointPillarsScatter(batch_size=config['batch_size'],
                                                               output_shape=config['grid_size'],
                                                               num_input_features=num_rpn_input_filters)

        self.rpn = RPN(num_rpn_input_filters)
        self.loss_generator = loss_generator

    def forward(self, example):
        voxels = torch.from_numpy(example["voxels"]).to(self.device)#.half()
        num_points_per_voxel = torch.from_numpy(example["num_points_per_voxel"]).to(self.device)
        coordinates = torch.from_numpy(example["coordinates"]).to(self.device)

        voxel_features = self.pillar_point_net(voxels, num_points_per_voxel, coordinates)
        spatial_features = self.middle_feature_extractor(voxel_features, coordinates)
        preds_dict = self.rpn(spatial_features)

        return preds_dict

