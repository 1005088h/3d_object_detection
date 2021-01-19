import torch
from torch import nn
from torch.nn import Sequential
import functools
import time
from framework.utils import change_default_args
from framework.trt_utils import export_onnx, build_engine, load_engine
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

pycuda.driver.Device(0).retain_primary_context()


### change anchor order, shared head, add trt

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
        x_max = x_max.squeeze(dim=0)
        return x_max


class PFN_trt:
    def __init__(self, engine_path):
        super().__init__()
        import tensorrt as trt
        engine = load_engine(engine_path)
        self.context = engine.create_execution_context()

        # dynamic shape, batch size
        self.context.active_optimization_profile = 0

        input_shape = (16000, 15, 9)
        self.context.set_binding_shape(0, input_shape)
        size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
        self.in_d = cuda.mem_alloc(size)
        self.out_h = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(1)), dtype=np.float32)
        self.out_d = cuda.mem_alloc(self.out_h.nbytes)

    def run(self, input):
        self.context.set_binding_shape(0, input.shape)
        output_shape = tuple(self.context.get_binding_shape(1))
        output = torch.empty(output_shape, dtype=torch.float32, device=torch.device("cuda:0"))
        self.context.execute_v2(bindings=[int(input.data_ptr()), int(output.data_ptr())])
        return output


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


class Scatter:
    def __init__(self,
                 output_shape,
                 num_input_features=64):
        super().__init__()
        self.name = 'PointPillarsScatter'
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
        canvas = canvas.contiguous()
        return canvas


class Scatter_cuda:
    def __init__(self, output_shape, num_input_features=64, max_num_pillars=16000):
        super().__init__()

        nx = output_shape[0]
        ny = output_shape[1]

        canvas_np = np.zeros([num_input_features, nx, ny]).astype(np.float32)
        self.canvas_size = canvas_np.size
        self.canvas_h = cuda.pagelocked_empty(canvas_np.size, dtype=np.float32)
        self.canvas_d = cuda.mem_alloc(self.canvas_h.nbytes)
        cuda.memset_d32(self.canvas_d, 0, self.canvas_size)

        stride_ca_h = np.zeros([2]).astype(np.int32)
        stride_ca_h[1] = canvas_np.shape[2]
        stride_ca_h[0] = canvas_np.shape[1] * stride_ca_h[1]
        self.stride_ca_d = cuda.mem_alloc(stride_ca_h.nbytes)
        cuda.memcpy_htod(self.stride_ca_d, stride_ca_h)

        # coors_h = coors.cpu().numpy()
        # coors_d = cuda.mem_alloc(coors_h.nbytes)
        # cuda.memcpy_htod(coors_d, coors_h)
        #
        # stride_vx_h = np.array(features.shape[0])
        # stride_vx_d = cuda.mem_alloc(stride_vx_h.nbytes)
        # cuda.memcpy_htod(stride_vx_d, stride_vx_h)

        coors_np = np.zeros([max_num_pillars, 3]).astype(np.int32)
        self.coors_d = cuda.mem_alloc(coors_np.nbytes)
        cuda.memset_d32(self.coors_d, 0, coors_np.size)

        stride_vx_h = np.zeros([1]).astype(np.int32)
        self.stride_vx_d = cuda.mem_alloc(stride_vx_h.nbytes)

        self.grid = (4, 250, 1)
        self.block = (16, 64, 1)

        mod = SourceModule("""
        __global__ void
        scatter(float *canvas, int *stride_ca, float *voxel, int *stride_vx, int *coors)
        {
            int idx_c = blockIdx.x * blockDim.x + threadIdx.x;
            int idx_o = blockIdx.y * blockDim.y + threadIdx.y;
        
            if(idx_o < stride_vx[0])
            {
                int cx = coors[idx_o * 3];
                int cy = coors[idx_o * 3 + 1];
        
                int idx_canvas = idx_c * stride_ca[0] + cx * stride_ca[1] + cy;
                int idx_voxel = idx_c * stride_vx[0] + idx_o;
                canvas[idx_canvas] = voxel[idx_voxel];
            }
        }
        """)

        self.scatter = mod.get_function("scatter")

    def run(self, voxel_d, coors_d):
        cuda.memset_d32(self.canvas_d, 0, self.canvas_size)
        self.scatter(self.canvas_d, self.stride_ca_d, voxel_d, self.stride_vx_d, coors_d, grid=self.grid,
                     block=self.block)
        return self.canvas_d

    def toHost(self, canvas_d):
        cuda.memcpy_dtoh(self.canvas_h, canvas_d)
        return self.canvas_h


class RPN_trt:
    def __init__(self, engine_path):
        super().__init__()
        import tensorrt as trt
        engine = load_engine(engine_path)
        self.context = engine.create_execution_context()
        self.out_shape = tuple(self.context.get_binding_shape(1))
        # self.out_h = cuda.pagelocked_empty(trt.volume(self.context.get_binding_shape(1)), dtype=np.float32)
        # self.out_d = cuda.mem_alloc(self.out_h.nbytes)
        #
        # self.rpn_context = engine.create_execution_context()

    def run(self, input):
        output = torch.empty(self.out_shape, dtype=torch.float32, device=torch.device("cuda:0"))
        self.context.execute_v2(bindings=[int(input.data_ptr()), int(output.data_ptr())])
        return output


class RPN(nn.Module):
    def __init__(self, num_rpn_input_filters):
        super().__init__()

        layer_strides = [2, 2, 2]
        num_filters = [64, 128, 256]
        upsample_strides = [1, 2, 4]
        num_upsample_filters = [64, 128, 128]  # [128, 128, 128]
        num_input_filters = num_rpn_input_filters

        self.out_plane = sum(num_upsample_filters)

        norm_layer = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        # norm_layer = change_default_args(eps=1e-3, momentum=0.01)(nn.InstanceNorm2d)
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


class SharedHead_trt:

    def __init__(self, engine_path, device=torch.device("cuda:0")):
        super().__init__()

        engine = load_engine(engine_path)
        self.context = engine.create_execution_context()

        cls_preds_shape = tuple(engine.get_binding_shape('cls_preds'))
        box_preds_shape = tuple(engine.get_binding_shape('box_preds'))
        dir_preds_shape = tuple(engine.get_binding_shape('dir_preds'))

        self.cls_preds = torch.empty(cls_preds_shape, dtype=torch.float32, device=device)
        self.box_preds = torch.empty(box_preds_shape, dtype=torch.float32, device=device)
        self.dir_preds = torch.empty(dir_preds_shape, dtype=torch.float32, device=device)

    def run(self, input):
        self.context.execute(bindings=[int(input.data_ptr()), int(self.cls_preds.data_ptr()),
                                       int(self.box_preds.data_ptr()), int(self.dir_preds.data_ptr())])
        return self.cls_preds, self.box_preds, self.dir_preds


class SharedHead(nn.Module):

    def __init__(self, in_plane, engine_path=None):
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


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


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

        self.pfn_time, self.scatter_time, self.rpn_time, self.heads_time = 0, 0, 0, 0

        pfn_engine_path = '../deployment/pfn16.engine'
        self.pfn = PFN_trt(pfn_engine_path)

        self.scatter = Scatter(output_shape=config['grid_size'], num_input_features=64)

        rpn_engine_path = '../deployment/rpn16.engine'
        self.rpn = RPN_trt(rpn_engine_path)

        head_engine_path = '../deployment/head16.engine'
        self.heads = SharedHead_trt(head_engine_path, device=self.device)

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

        # start network
        start = time.time()
        pfn_out = self.pfn.run(features)
        torch.cuda.synchronize()
        pfn_time = time.time()

        voxel_feature = self.scatter.run(pfn_out, coors)
        torch.cuda.synchronize()
        scatter_time = time.time()

        rpn_out = self.rpn.run(voxel_feature)

        torch.cuda.synchronize()
        rpn_time = time.time()

        cls_preds, box_preds, dir_preds = self.heads.run(rpn_out)

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

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
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

        voxel_features = self.pillar_point_net(features)
        voxel_features = voxel_features.squeeze()
        pfn_onnx_file_path = '../deployment/pfn.onnx'
        pfn_engine_path = '../deployment/pfn16.engine'
        # export(self.pillar_point_net, features, pfn_onnx_file_path, dynamic=True)
        # build_engine(pfn_onnx_file_path, pfn_engine_path, dynamic=True)
        # exit(0)

        # pfn_context = load_engine_context(pfn_engine_path)
        # h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        # h_output = self.pfn_infer(pfn_context, h_input)
        # voxel_features = torch.Tensor(h_output).view(features.shape[0], -1)
        # voxel_features_np = voxel_features.cpu().numpy()

        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        rpn_onnx_file_path = '../deployment/rpn_inst2.onnx'
        rpn_engine_path = '../deployment/rpn_inst2.engine'

        rpn_feature = self.rpn(spatial_features)
        export(self.rpn, spatial_features, rpn_onnx_file_path, dynamic=False)
        build_engine(rpn_onnx_file_path, rpn_engine_path, dynamic=False)
        exit(0)

        # preds_dict = self.heads(rpn_feature)

        cls_preds, box_preds, dir_preds = self.heads(rpn_feature)

        head_onnx_file_path = '../deployment/head.onnx'
        head_engine_path = '../deployment/head16.engine'

        export(self.heads, rpn_feature, head_onnx_file_path, dynamic=False, input_names=['inputs'],
               output_names=['cls_preds', 'box_preds', 'dir_preds'])
        build_engine(head_onnx_file_path, head_engine_path, dynamic=False)
        exit(0)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        # self.voxel_features_time += voxel_features_time - start
        # self.spatial_features_time += spatial_features_time - voxel_features_time
        # self.rpn_feature_time += rpn_feature_time - spatial_features_time
        # self.heads_time += heads_time - rpn_feature_time

        return preds_dict


class PointPillars_v3(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        voxel_size = config['voxel_size']
        offset = config['detection_offset']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        self.pfn_time, self.scatter_time, self.rpn_time, self.heads_time = 0, 0, 0, 0

        pfn_engine_path = '../deployment/pfn16.engine'
        self.pfn = PFN_trt(pfn_engine_path)

        self.scatter = Scatter(output_shape=config['grid_size'], num_input_features=64)
        # self.scatter = Scatter_cuda(output_shape=config['grid_size'], num_input_features=64)

        rpn_engine_path = '../deployment/rpn16.engine'
        self.rpn = RPN_trt(rpn_engine_path)

        head_engine_path = '../deployment/head16.engine'
        self.heads = SharedHead_trt(head_engine_path, device=self.device)

    def forward(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

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
        pfn_out = self.pfn.run(features)
        torch.cuda.synchronize()
        pfn_time = time.time()

        # stride_vx_h = np.array(features.shape[0]).astype(np.int32)
        # cuda.memcpy_htod(self.scatter.stride_vx_d, stride_vx_h)
        # voxel_feature_d = self.scatter.run(Holder(pfn_out), Holder(coors))
        voxel_feature = self.scatter.run(pfn_out, coors)
        torch.cuda.synchronize()
        scatter_time = time.time()

        # self.rpn.context.execute(bindings=[int(voxel_feature_d), int(self.rpn.out_d)])
        rpn_out = self.rpn.run(voxel_feature)
        # self.rpn.context.execute(bindings=[int(voxel_feature_d.data_ptr()), int(self.rpn.out_d)])
        torch.cuda.synchronize()
        rpn_time = time.time()

        cls_preds, box_preds, dir_preds = self.heads.run(rpn_out)
        # self.heads.context.execute(
        #     bindings=[int(rpn_out.data_ptr()), int(self.heads.cls_preds_d.data_ptr()), int(self.heads.box_preds_d.data_ptr()),
        #               int(self.heads.dir_preds_d.data_ptr())])
        torch.cuda.synchronize()
        heads_time = time.time()
        #
        # cls_preds = self.heads.cls_preds_d.view(1, -1, 1)
        # box_preds = self.heads.box_preds_d.view(1, -1, 7)
        # dir_preds = self.heads.dir_preds_d.view(1, -1, 2)

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

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
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

        voxel_features = self.pillar_point_net(features)
        voxel_features = voxel_features.squeeze()
        pfn_onnx_file_path = '../deployment/pfn.onnx'
        pfn_engine_path = '../deployment/pfn16.engine'
        # export(self.pillar_point_net, features, pfn_onnx_file_path, dynamic=True)
        # build_engine(pfn_onnx_file_path, pfn_engine_path, dynamic=True)
        # exit(0)

        # pfn_context = load_engine_context(pfn_engine_path)
        # h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        # h_output = self.pfn_infer(pfn_context, h_input)
        # voxel_features = torch.Tensor(h_output).view(features.shape[0], -1)
        # voxel_features_np = voxel_features.cpu().numpy()

        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        rpn_onnx_file_path = '../deployment/rpn_inst2.onnx'
        rpn_engine_path = '../deployment/rpn_inst2.engine'

        rpn_feature = self.rpn(spatial_features)
        export(self.rpn, spatial_features, rpn_onnx_file_path, dynamic=False)
        build_engine(rpn_onnx_file_path, rpn_engine_path, dynamic=False)
        exit(0)

        # preds_dict = self.heads(rpn_feature)

        cls_preds, box_preds, dir_preds = self.heads(rpn_feature)

        head_onnx_file_path = '../deployment/head.onnx'
        head_engine_path = '../deployment/head16.engine'

        export(self.heads, rpn_feature, head_onnx_file_path, dynamic=False, input_names=['inputs'],
               output_names=['cls_preds', 'box_preds', 'dir_preds'])
        build_engine(head_onnx_file_path, head_engine_path, dynamic=False)
        exit(0)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        # self.voxel_features_time += voxel_features_time - start
        # self.spatial_features_time += spatial_features_time - voxel_features_time
        # self.rpn_feature_time += rpn_feature_time - spatial_features_time
        # self.heads_time += heads_time - rpn_feature_time

        return preds_dict


class PointPillars_v2(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        voxel_size = config['voxel_size']
        offset = config['detection_offset']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        self.pfn_time, self.scatter_time, self.rpn_time, self.heads_time = 0, 0, 0, 0

        pfn_engine_path = '../deployment/pfn16.engine'
        self.pfn = PFN_trt(pfn_engine_path)

        self.scatter = Scatter_cuda(output_shape=config['grid_size'], num_input_features=64)

        rpn_engine_path = '../deployment/rpn16.engine'
        self.rpn = RPN_trt(rpn_engine_path)

        head_engine_path = '../deployment/head16.engine'
        self.heads = SharedHead_trt(head_engine_path)

    def forward(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

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
        # pfn_in_h = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        self.pfn.context.set_binding_shape(0, features.shape)
        # cuda.memcpy_htod(self.pfn.in_d, pfn_in_h)
        self.pfn.context.execute_v2(bindings=[int(features.data_ptr()), int(self.pfn.out_d)])
        pfn_time = time.time()

        # coors_h = coors.cpu().numpy().astype(np.int32)
        # cuda.memcpy_htod(self.scatter.coors_d, coors_h)

        stride_vx_h = np.array(features.shape[0]).astype(np.int32)
        cuda.memcpy_htod(self.scatter.stride_vx_d, stride_vx_h)
        voxel_feature_d = self.scatter.run(self.pfn.out_d, Holder(coors))
        scatter_time = time.time()

        self.rpn.context.execute(bindings=[int(voxel_feature_d), int(self.rpn.out_d)])
        rpn_time = time.time()

        self.heads.context.execute(
            bindings=[int(self.rpn.out_d), int(self.heads.cls_preds_d), int(self.heads.box_preds_d),
                      int(self.heads.dir_preds_d)])
        heads_time = time.time()

        cuda.memcpy_dtoh(self.heads.cls_preds_h, self.heads.cls_preds_d)
        cuda.memcpy_dtoh(self.heads.box_preds_h, self.heads.box_preds_d)
        cuda.memcpy_dtoh(self.heads.dir_preds_h, self.heads.dir_preds_d)

        cls_preds = torch.from_numpy(self.heads.cls_preds_h).to(self.device).view(1, -1, 1)
        box_preds = torch.from_numpy(self.heads.box_preds_h).to(self.device).view(1, -1, 7)
        dir_preds = torch.from_numpy(self.heads.dir_preds_h).to(self.device).view(1, -1, 2)

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

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
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

        voxel_features = self.pillar_point_net(features)
        voxel_features = voxel_features.squeeze()
        pfn_onnx_file_path = '../deployment/pfn.onnx'
        pfn_engine_path = '../deployment/pfn16.engine'
        # export(self.pillar_point_net, features, pfn_onnx_file_path, dynamic=True)
        # build_engine(pfn_onnx_file_path, pfn_engine_path, dynamic=True)
        # exit(0)

        # pfn_context = load_engine_context(pfn_engine_path)
        # h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        # h_output = self.pfn_infer(pfn_context, h_input)
        # voxel_features = torch.Tensor(h_output).view(features.shape[0], -1)
        # voxel_features_np = voxel_features.cpu().numpy()

        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        rpn_onnx_file_path = '../deployment/rpn_inst2.onnx'
        rpn_engine_path = '../deployment/rpn_inst2.engine'

        rpn_feature = self.rpn(spatial_features)
        export(self.rpn, spatial_features, rpn_onnx_file_path, dynamic=False)
        build_engine(rpn_onnx_file_path, rpn_engine_path, dynamic=False)
        exit(0)

        # preds_dict = self.heads(rpn_feature)

        cls_preds, box_preds, dir_preds = self.heads(rpn_feature)

        head_onnx_file_path = '../deployment/head.onnx'
        head_engine_path = '../deployment/head16.engine'

        export(self.heads, rpn_feature, head_onnx_file_path, dynamic=False, input_names=['inputs'],
               output_names=['cls_preds', 'box_preds', 'dir_preds'])
        build_engine(head_onnx_file_path, head_engine_path, dynamic=False)
        exit(0)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        # self.voxel_features_time += voxel_features_time - start
        # self.spatial_features_time += spatial_features_time - voxel_features_time
        # self.rpn_feature_time += rpn_feature_time - spatial_features_time
        # self.heads_time += heads_time - rpn_feature_time

        return preds_dict


class PointPillars_v1(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config['device']
        # self.pillar_point_net = PointNet()
        # num_rpn_input_filters = self.pillar_point_net.out_channels
        # self.middle_feature_extractor = PointPillarsScatter_v2(batch_size=config['batch_size'],
        #                                                        output_shape=config['grid_size'],
        #                                                        num_input_features=num_rpn_input_filters)
        #
        #
        # self.rpn = RPN(num_rpn_input_filters)
        # self.heads = SharedHead_trt(320)
        # self.pfn_time, self.rpn_time, self.scatter_time, self.heads_time = 0.0, 0.0, 0.0, 0.0
        #
        voxel_size = config['voxel_size']
        offset = config['detection_offset']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        # pfn_engine_path = '../deployment/pfn16.engine'
        # self.pfn = PFN_trt(pfn_engine_path)
        #
        self.scatter = Scatter_cuda(output_shape=config['grid_size'], num_input_features=64)
        #
        # rpn_engine_path = '../deployment/rpn16.engine'
        # self.rpn = RPN_trt(rpn_engine_path)
        #
        head_engine_path = '../deployment/head16.engine'
        self.heads = SharedHead_trt(head_engine_path)

        import tensorrt as trt
        pfn_engine_path = '../deployment/pfn16.engine'
        engine = load_engine(pfn_engine_path)
        self.pfn_context = engine.create_execution_context()

        # dynamic shape, batch size
        self.pfn_context.active_optimization_profile = 0

        pfn_input_shape = (16000, 15, 9)
        self.pfn_context.set_binding_shape(0, pfn_input_shape)
        size = trt.volume(pfn_input_shape) * np.dtype(np.float32).itemsize
        self.pfn_i_d = cuda.mem_alloc(size)
        self.pfn_o_h = cuda.pagelocked_empty(trt.volume(self.pfn_context.get_binding_shape(1)), dtype=np.float32)
        self.pfn_o_d = cuda.mem_alloc(self.pfn_o_h.nbytes)

        rpn_engine_path = '../deployment/rpn16.engine'
        engine = load_engine(rpn_engine_path)
        self.rpn_context = engine.create_execution_context()

        rpn_input_shape = (1, 64, 800, 800)
        self.rpn_context.set_binding_shape(0, rpn_input_shape)
        # size = trt.volume(rpn_input_shape) * np.dtype(np.float32).itemsize
        # self.d_input = cuda.mem_alloc(size)
        self.rpn_out_h = cuda.pagelocked_empty(trt.volume(self.rpn_context.get_binding_shape(1)), dtype=np.float32)
        self.rpn_out_d = cuda.mem_alloc(self.rpn_out_h.nbytes)

    def forward(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

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

        h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        self.pfn_context.set_binding_shape(0, h_input.shape)

        cuda.memcpy_htod(self.pfn_i_d, h_input)

        coors_h = coors.cpu().numpy()
        coors_d = cuda.mem_alloc(coors_h.nbytes)
        cuda.memcpy_htod(coors_d, coors_h)

        stride_vx_h = np.array(features.shape[0])
        stride_vx_d = cuda.mem_alloc(stride_vx_h.nbytes)
        cuda.memcpy_htod(stride_vx_d, stride_vx_h)
        start = time.time()
        self.pfn_context.execute_v2(bindings=[int(self.pfn_i_d), int(self.pfn_o_d)])
        pfn_time = time.time()
        self.scatter.clear()
        spatial_features_cuda_d = self.scatter.run(self.pfn_o_d, coors_d, stride_vx_d)
        scatter_time = time.time()

        self.rpn_context.execute(bindings=[int(spatial_features_cuda_d), int(self.rpn_out_d)])
        rpn_time = time.time()
        self.heads.context.execute(
            bindings=[int(self.rpn_out_d), int(self.heads.cls_preds_d), int(self.heads.box_preds_d),
                      int(self.heads.dir_preds_d)])
        heads_time = time.time()

        cuda.memcpy_dtoh(self.heads.cls_preds_h, self.heads.cls_preds_d)
        cuda.memcpy_dtoh(self.heads.box_preds_h, self.heads.box_preds_d)
        cuda.memcpy_dtoh(self.heads.dir_preds_h, self.heads.dir_preds_d)

        cls_preds = torch.from_numpy(self.heads.cls_preds_h).to(self.device).view(1, -1, 1)
        box_preds = torch.from_numpy(self.heads.box_preds_h).to(self.device).view(1, -1, 7)
        dir_preds = torch.from_numpy(self.heads.dir_preds_h).to(self.device).view(1, -1, 2)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        # self.pfn_time += pfn_time - start
        # self.scatter_time += scatter_time - pfn_time
        # self.rpn_time += rpn_time - scatter_time
        # self.heads_time += heads_time - rpn_time

        return preds_dict

    def export(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
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

        voxel_features = self.pillar_point_net(features)
        voxel_features = voxel_features.squeeze()
        pfn_onnx_file_path = '../deployment/pfn.onnx'
        pfn_engine_path = '../deployment/pfn16.engine'
        # export(self.pillar_point_net, features, pfn_onnx_file_path, dynamic=True)
        # build_engine(pfn_onnx_file_path, pfn_engine_path, dynamic=True)
        # exit(0)

        # pfn_context = load_engine_context(pfn_engine_path)
        # h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        # h_output = self.pfn_infer(pfn_context, h_input)
        # voxel_features = torch.Tensor(h_output).view(features.shape[0], -1)
        # voxel_features_np = voxel_features.cpu().numpy()

        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        rpn_onnx_file_path = '../deployment/rpn_inst2.onnx'
        rpn_engine_path = '../deployment/rpn_inst2.engine'

        rpn_feature = self.rpn(spatial_features)
        export(self.rpn, spatial_features, rpn_onnx_file_path, dynamic=False)
        build_engine(rpn_onnx_file_path, rpn_engine_path, dynamic=False)
        exit(0)

        # preds_dict = self.heads(rpn_feature)

        cls_preds, box_preds, dir_preds = self.heads(rpn_feature)

        head_onnx_file_path = '../deployment/head.onnx'
        head_engine_path = '../deployment/head16.engine'

        export(self.heads, rpn_feature, head_onnx_file_path, dynamic=False, input_names=['inputs'],
               output_names=['cls_preds', 'box_preds', 'dir_preds'])
        build_engine(head_onnx_file_path, head_engine_path, dynamic=False)
        exit(0)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        # self.voxel_features_time += voxel_features_time - start
        # self.spatial_features_time += spatial_features_time - voxel_features_time
        # self.rpn_feature_time += rpn_feature_time - spatial_features_time
        # self.heads_time += heads_time - rpn_feature_time

        return preds_dict


class PointPillars_v0(nn.Module):

    def __init__(self, config):
        super().__init__()

        voxel_size = config['voxel_size']
        offset = config['detection_offset']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        pfn_engine_path = '../deployment/pfn16.engine'
        self.pfn = PFN_trt(pfn_engine_path)

        self.scatter = Scatter_cuda(output_shape=config['grid_size'], num_input_features=64)

        rpn_engine_path = '../deployment/rpn16.engine'
        self.rpn = RPN_trt(rpn_engine_path)

        head_engine_path = '../deployment/head16.engine'
        self.heads = SharedHead_trt(head_engine_path)

    def forward_new(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

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

        pillars_h = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        cuda.memcpy_htod(self.pfn.in_d, pillars_h)
        voxels_d = self.pfn.run(pillars_h.shape)

        coors_h = coors.cpu().numpy()
        coors_d = cuda.mem_alloc(coors_h.nbytes)
        cuda.memcpy_htod(coors_d, coors_h)

        stride_vx_h = np.array(features.shape[0])
        stride_vx_d = cuda.mem_alloc(stride_vx_h.nbytes)

        cuda.memcpy_htod(stride_vx_d, stride_vx_h)
        self.scatter.clear()
        spatial_features_cuda_d = self.scatter.run(voxels_d, coors_d, stride_vx_d)

        rpn_out_d = self.rpn.run(canvas_d)
        cls_preds_d, box_preds_d, dir_preds_d = self.heads.run(rpn_out_d)

        cuda.memcpy_dtoh(self.heads.cls_preds_h, cls_preds_d)
        cuda.memcpy_dtoh(self.heads.box_preds_h, box_preds_d)
        cuda.memcpy_dtoh(self.heads.dir_preds_h, dir_preds_d)

        cls_preds = torch.from_numpy(self.heads.cls_preds_h).to(self.device).view(1, -1, 1)
        box_preds = torch.from_numpy(self.heads.box_preds_h).to(self.device).view(1, -1, 7)
        dir_preds = torch.from_numpy(self.heads.dir_preds_h).to(self.device).view(1, -1, 2)

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

    def forward(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
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
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # start = time.time()
        # voxel_features = self.pillar_point_net(features)
        # torch.cuda.synchronize()

        h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        self.pfn_context.set_binding_shape(0, h_input.shape)
        stream = cuda.Stream()
        cuda.memcpy_htod_async(self.pfn_i_d, h_input, stream)

        start = time.time()
        self.pfn_context.execute_async_v2(bindings=[int(self.pfn_i_d), int(self.pfn_o_d)], stream_handle=stream.handle)
        stream.synchronize()
        voxel_features_time = time.time()

        # cuda.memcpy_dtoh_async(self.pfn_o_h, self.pfn_o_d, stream)
        # stream.synchronize()
        # size = 64 * features.shape[0]
        # # h_output = self.h_output.reshape(64, -1)
        # h_output = self.h_output
        # voxel_features = torch.Tensor(h_output).cuda()

        # spatial_features = self.middle_feature_extractor(voxel_features, coors)

        # voxel_h = voxel_features.cpu().numpy()
        # voxel_d = cuda.mem_alloc(voxel_h.nbytes)
        # cuda.memcpy_htod(voxel_d, voxel_h)

        voxel_d = self.pfn_o_d
        coors_h = coors.cpu().numpy()
        coors_d = cuda.mem_alloc(coors_h.nbytes)
        cuda.memcpy_htod(coors_d, coors_h)

        stride_vx_h = np.array(features.shape[0])
        stride_vx_d = cuda.mem_alloc(stride_vx_h.nbytes)
        cuda.memcpy_htod(stride_vx_d, stride_vx_h)
        self.scatter.reset()
        torch.cuda.synchronize()
        voxel_features_time = time.time()
        spatial_features_cuda_d = self.middle_feature_extractor_cuda.run(voxel_d, coors_d, stride_vx_d)
        # torch.cuda.synchronize()

        # cuda.memcpy_dtoh(self.middle_feature_extractor_cuda.canvas_h, spatial_features_cuda_d)
        # spatial_features = torch.from_numpy(self.middle_feature_extractor_cuda.canvas_h).to(self.device).view(1, 64,
        #                                                                                                       800, 800)

        # spatial_features_np = np.ravel(spatial_features.cpu().numpy())
        # diff = spatial_features_cuda - spatial_features_np
        # diff = np.fabs(diff) > 0
        # print(diff.any())
        # diff[0] = 0.1
        # diff = np.fabs(diff) > 0
        # print(diff.any())
        # torch.cuda.synchronize()

        # rpn_feature = self.rpn(spatial_features)
        spatial_features_time = time.time()
        self.rpn_context.execute_async_v2(bindings=[int(spatial_features_cuda_d), int(self.rpn_out_d)],
                                          stream_handle=stream.handle)
        stream.synchronize()

        cuda.memcpy_dtoh_async(self.rpn_out_h, self.rpn_out_d, stream)
        rpn_feature_h = self.rpn_out_h.reshape(self.rpn_context.get_binding_shape(1))
        rpn_feature = torch.from_numpy(rpn_feature_h).to(self.device)
        # rpn_feature_np = rpn_feature.cpu().numpy()
        # diff = rpn_feature_h - rpn_feature_np
        # diff = np.fabs(diff) > 1e-4
        # print(diff.any())

        torch.cuda.synchronize()

        # cls_preds, box_preds, dir_preds = self.heads(rpn_feature)
        rpn_feature_time = time.time()

        self.heads.head_context.execute_async_v2(
            bindings=[int(self.rpn_out_d), int(self.heads.cls_preds_d_o), int(self.heads.box_preds_d_o),
                      int(self.heads.dir_preds_d_o)],
            stream_handle=stream.handle)
        stream.synchronize()
        heads_time = time.time()
        cuda.memcpy_dtoh_async(self.heads.cls_preds_h_o, self.heads.cls_preds_d_o, stream)
        cuda.memcpy_dtoh_async(self.heads.box_preds_h_o, self.heads.box_preds_d_o, stream)
        cuda.memcpy_dtoh_async(self.heads.dir_preds_h_o, self.heads.dir_preds_d_o, stream)
        stream.synchronize()

        cls_preds = torch.from_numpy(self.heads.cls_preds_h_o).to(self.device).view(1, -1, 1)
        box_preds = torch.from_numpy(self.heads.box_preds_h_o).to(self.device).view(1, -1, 7)
        dir_preds = torch.from_numpy(self.heads.dir_preds_h_o).to(self.device).view(1, -1, 2)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }
        torch.cuda.synchronize()

        self.voxel_features_time += voxel_features_time - start
        self.spatial_features_time += spatial_features_time - voxel_features_time
        self.rpn_feature_time += rpn_feature_time - spatial_features_time
        self.heads_time += heads_time - rpn_feature_time

        return preds_dict

    def export(self, example):
        voxels = example["voxels"]
        num_point_per_voxel = example["num_points_per_voxel"]
        coors = example["coordinates"]

        # Find distance of x, y, and z from cluster center
        points_mean = voxels[:, :, :3].sum(dim=1, keepdim=True) / num_point_per_voxel.type_as(voxels).view(-1, 1, 1)
        f_cluster = voxels[:, :, :3] - points_mean

        # Find distance of x and y from pillar center
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

        voxel_features = self.pillar_point_net(features)
        voxel_features = voxel_features.squeeze()
        pfn_onnx_file_path = '../deployment/pfn.onnx'
        pfn_engine_path = '../deployment/pfn16.engine'
        # export(self.pillar_point_net, features, pfn_onnx_file_path, dynamic=True)
        # build_engine(pfn_onnx_file_path, pfn_engine_path, dynamic=True)
        # exit(0)

        # pfn_context = load_engine_context(pfn_engine_path)
        # h_input = np.array(features.cpu().numpy(), dtype=np.float32, order='C')
        # h_output = self.pfn_infer(pfn_context, h_input)
        # voxel_features = torch.Tensor(h_output).view(features.shape[0], -1)
        # voxel_features_np = voxel_features.cpu().numpy()

        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        rpn_onnx_file_path = '../deployment/rpn_inst2.onnx'
        rpn_engine_path = '../deployment/rpn_inst2.engine'

        rpn_feature = self.rpn(spatial_features)
        export(self.rpn, spatial_features, rpn_onnx_file_path, dynamic=False)
        build_engine(rpn_onnx_file_path, rpn_engine_path, dynamic=False)
        exit(0)

        # preds_dict = self.heads(rpn_feature)

        cls_preds, box_preds, dir_preds = self.heads(rpn_feature)

        head_onnx_file_path = '../deployment/head.onnx'
        head_engine_path = '../deployment/head16.engine'

        export(self.heads, rpn_feature, head_onnx_file_path, dynamic=False, input_names=['inputs'],
               output_names=['cls_preds', 'box_preds', 'dir_preds'])
        build_engine(head_onnx_file_path, head_engine_path, dynamic=False)
        exit(0)

        preds_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        # self.voxel_features_time += voxel_features_time - start
        # self.spatial_features_time += spatial_features_time - voxel_features_time
        # self.rpn_feature_time += rpn_feature_time - spatial_features_time
        # self.heads_time += heads_time - rpn_feature_time

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


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()
