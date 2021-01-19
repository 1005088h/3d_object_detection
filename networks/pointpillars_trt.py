import torch
from torch import nn
from framework.trt_utils import load_engine
import pycuda.driver as cuda
import pycuda.autoinit

pycuda.driver.Device(0).retain_primary_context()


class PFN:
    def __init__(self, engine_path):
        super().__init__()
        engine = load_engine(engine_path)
        self.context = engine.create_execution_context()
        # dynamic shape, batch size
        self.context.active_optimization_profile = 0

    def run(self, input):
        self.context.set_binding_shape(0, input.shape)
        output_shape = tuple(self.context.get_binding_shape(1))
        output = torch.empty(output_shape, dtype=torch.float32, device=input.device)
        self.context.execute_v2(bindings=[int(input.data_ptr()), int(output.data_ptr())])
        return output


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


class RPN:
    def __init__(self, engine_path):
        super().__init__()
        engine = load_engine(engine_path)
        self.context = engine.create_execution_context()
        self.out_shape = tuple(self.context.get_binding_shape(1))

    def run(self, input):
        output = torch.empty(self.out_shape, dtype=torch.float32, device=input.device)
        self.context.execute_v2(bindings=[int(input.data_ptr()), int(output.data_ptr())])
        return output


class SharedHead:
    def __init__(self, engine_path):
        super().__init__()

        engine = load_engine(engine_path)
        self.context = engine.create_execution_context()

        self.cls_preds_shape = tuple(engine.get_binding_shape('cls_preds'))
        self.box_preds_shape = tuple(engine.get_binding_shape('box_preds'))
        self.dir_preds_shape = tuple(engine.get_binding_shape('dir_preds'))

    def run(self, input):
        cls_preds = torch.empty(self.cls_preds_shape, dtype=torch.float32, device=input.device)
        box_preds = torch.empty(self.box_preds_shape, dtype=torch.float32, device=input.device)
        dir_preds = torch.empty(self.dir_preds_shape, dtype=torch.float32, device=input.device)
        self.context.execute(bindings=[int(input.data_ptr()), int(cls_preds.data_ptr()),
                                       int(box_preds.data_ptr()), int(dir_preds.data_ptr())])
        return cls_preds, box_preds, dir_preds


class PointPillars(nn.Module):

    def __init__(self, config, pfn_engine_path, rpn_engine_path, head_engine_path):
        super().__init__()
        self.device = config['device']
        voxel_size = config['voxel_size']
        offset = config['detection_offset']
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + offset[0]
        self.y_offset = self.vy / 2 + offset[1]

        # self.pfn_time, self.scatter_time, self.rpn_time, self.heads_time = 0, 0, 0, 0
        self.pfn = PFN(pfn_engine_path)
        self.scatter = Scatter(output_shape=config['grid_size'], num_input_features=64)
        self.rpn = RPN(rpn_engine_path)
        self.heads = SharedHead(head_engine_path)

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
        # start = time.time()
        pfn_out = self.pfn.run(features)
        # torch.cuda.synchronize()
        # pfn_time = time.time()

        voxel_feature = self.scatter.run(pfn_out, coors)
        # torch.cuda.synchronize()
        # scatter_time = time.time()

        rpn_out = self.rpn.run(voxel_feature)
        # torch.cuda.synchronize()
        # rpn_time = time.time()

        cls_preds, box_preds, dir_preds = self.heads.run(rpn_out)
        # torch.cuda.synchronize()
        # heads_time = time.time()

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