from torch.utils.data import Dataset
import torch
import pickle
import numpy as np
import framework.box_np_ops as box_np_ops
import time
from pathlib import Path
import matplotlib.pyplot as plt
from framework import augmentation as agm

class GenericDataset(Dataset):
    def __init__(self, config, info_path, voxel_generator, anchor_assigner, training=True):
        with open(info_path, 'rb') as f:
            self.infos = pickle.load(f)
        self.root_dir = Path(info_path).parent
        self.num_point_features = config['num_point_features']
        self.voxel_generator = voxel_generator
        self.anchor_assigner = anchor_assigner
        self.detect_class = config['detect_class']
        self.detection_range = config['detection_range']
        self.grid_size = config['grid_size']
        self.training = training
        self.voxelization_t = 0.0
        self.load_t = 0.0

        car_total = 0
        truck_total = 0
        bus_total = 0
        car_dim = np.zeros(3)
        truck_dim = np.zeros(3)
        bus_dim = np.zeros(3)

        H, L = [], []
        false_box_count = 0
        for idx, info in enumerate(self.infos):
            if len(info['annos']['name']) > 0:
                car_mask = info['annos']['name'] == 'car'
                dims = info['annos']["dimensions"][car_mask]
                car_dim += np.sum(dims, axis=0)
                car_total += car_mask.sum()

                truck_mask = info['annos']['name'] == 'truck'
                dims = info['annos']["dimensions"][truck_mask]
                truck_dim += np.sum(dims, axis=0)
                truck_total += truck_mask.sum()

                bus_mask = info['annos']['name'] == 'bus'
                dims = info['annos']["dimensions"][bus_mask]
                bus_dim += np.sum(dims, axis=0)
                bus_total += bus_mask.sum()

                class_mask = car_mask | truck_mask | bus_mask
                info['annos']['name'][class_mask] = "vehicle"
                locs = info['annos']["location"][class_mask]
                dims = info['annos']["dimensions"][class_mask]
                H.append(locs[:, 2] + dims[:, 2] / 2)
                L.append(locs[:, 2] - dims[:, 2] / 2)

                difficulty_mask = info['annos']["difficulty"] > 0
                for key in info['annos']:
                    info['annos'][key] = info['annos'][key][difficulty_mask]

                num_mask = info['annos']["num_points"] == 0
                #class_mask = info['annos']['name'] == "vehicle"
                #num_mask = class_mask & num_mask
                if num_mask.sum() > 0:
                    false_box_count += num_mask.sum()
                    #info['annos']['name'][num_mask] = "ignore"
                    print("False empty boxes in frame: %d" % idx)
        print("Total false empty boxes : %d" % false_box_count)
        H = np.concatenate(H)
        L = np.concatenate(L)
        bus_dim = bus_dim / bus_total
        car_dim = car_dim / car_total
        truck_dim = truck_dim / truck_total

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]
        # read input
        t = time.time()
        v_path = self.root_dir / info['velodyne_path']
        points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, self.num_point_features])
        self.load_t += (time.time() - t)

        # read calib
        rect = info['calib/R0_rect'].astype(np.float32)
        Trv2c = info['calib/Tr_velo_to_cam'].astype(np.float32)
        P2 = info['calib/P2'].astype(np.float32)
        example = {
            'image_idx': info["image_idx"],
            'image_shape': info["img_shape"],
            'rect': rect,
            'Trv2c': Trv2c,
            'P2': P2,
        }
        # read ground truth if training
        if self.training:
            annos = info['annos']
            # filter class
            gt_class_mask = np.array([n in self.detect_class for n in annos["name"]], dtype=np.bool_)
            gt_names = annos["name"][gt_class_mask]
            gt_classes = np.array([self.detect_class.index(n) + 1 for n in gt_names], dtype=np.int32)
            loc = annos["location"][gt_class_mask]
            dims = annos["dimensions"][gt_class_mask]
            rots = annos["rotation_y"][gt_class_mask]
            difficulty = annos["difficulty"][gt_class_mask]
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            # data augmentation
            gt_boxes, points = agm.random_flip(gt_boxes, points)
            gt_boxes, points = agm.global_rotation(gt_boxes, points, rotation=np.pi / 8)
            gt_boxes, points = agm.global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05)
            gt_boxes, points = agm.global_translate(gt_boxes, points, noise_translate_std=[0.25, 0.25, 0.15])

            # filter range
            bv_range = self.detection_range[[0, 1, 3, 4]]
            range_mask = box_np_ops.filter_gt_box_outside_range(gt_boxes, bv_range)
            gt_boxes = gt_boxes[range_mask]
            gt_classes = gt_classes[range_mask]
            gt_names = gt_names[range_mask]
            difficulty = difficulty[range_mask]
            gt_boxes[:, 6] = box_np_ops.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

            example['annos'] = {'gt_classes': gt_classes, 'gt_boxes': gt_boxes, 'difficulty': difficulty, 'gt_names': gt_names}

            np.random.shuffle(points)
        self.points = points
        t = time.time()
        voxels, coors, num_points_per_voxel = self.voxel_generator.generate(points)
        '''
        count = 0
        for num in num_points_per_voxel:
            if num < 30:
                count += 1
        print(count / len(num_points_per_voxel))
        '''
        grid_size = self.grid_size
        voxel_size = self.voxel_generator.voxel_size
        offset = self.voxel_generator.offset
        anchors_mask = self.anchor_assigner.create_mask(coors, grid_size, voxel_size, offset)
        example['voxels'] = voxels
        example['coordinates'] = coors
        example['num_points_per_voxel'] = num_points_per_voxel
        example['anchors_mask'] = anchors_mask
        example['points'] = points
        self.voxelization_t += (time.time() - t)

        if self.training:
            gt_classes = example['annos']['gt_classes']
            gt_boxes = example['annos']['gt_boxes']
            label, bbox_targets, bbox_outside_weights, dir_cls_targets = self.anchor_assigner.assign(gt_classes,
                                                                                                     gt_boxes,
                                                                                                     anchors_mask)

            example['labels'] = label
            example['bbox_targets'] = bbox_targets
            example['dir_cls_targets'] = dir_cls_targets
            example['bbox_outside_weights'] = bbox_outside_weights

        return example
