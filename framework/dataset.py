from torch.utils.data import Dataset
import torch
import pickle
import numpy as np
import framework.box_np_ops as box_np_ops
import time
from pathlib import Path
import matplotlib.pyplot as plt
from framework import augmentation as agm
from framework.utils import example_convert_to_torch


class GenericDataset(Dataset):
    def __init__(self, config, info_paths, voxel_generator, anchor_assigner, training=True, augm=True):

        self.data_root = Path(config['data_root'])
        self.infos = []
        for info_path in info_paths:
            print(info_path)
            info_path = self.data_root / info_path
            with open(info_path, 'rb') as f:
                self.infos += pickle.load(f)
                print(len(self.infos))
        self.num_point_features = config['num_point_features']
        self.voxel_generator = voxel_generator
        self.anchor_assigner = anchor_assigner

        self.detect_class = config['detect_class']
        self.augm_class = config['detect_class']

        self.detection_range = config['detection_range']
        self.grid_size = config['grid_size']
        self.training = training
        self.augm = augm
        self.voxelization_t = 0.0
        self.load_t = 0.0
        self.create_mask_gpu = config['create_mask_gpu'] == 1
        car_total = 0
        truck_total = 0
        bus_total = 0
        person_total = 0
        motorbike_total = 0
        bicycle_total = 0

        for idx, info in enumerate(self.infos):
            if len(info['annos']['name']) > 0:

                difficulty_mask = info['annos']["num_points"] > 0
                for key in info['annos']:
                    info['annos'][key] = info['annos'][key][difficulty_mask]

                car_mask = info['annos']['name'] == 'car'
                car_total += car_mask.sum()

                truck_mask = info['annos']['name'] == 'truck'
                truck_total += truck_mask.sum()

                bus_mask = info['annos']['name'] == 'bus'
                bus_total += bus_mask.sum()

                person_mask = info['annos']['name'] == 'person'
                person_total += person_mask.sum()

                bicycle_mask = info['annos']['name'] == 'bicycle'
                bicycle_total += bicycle_mask.sum()

                motorbike_mask = info['annos']['name'] == 'motorbike'
                motorbike_total += motorbike_mask.sum()

                vehicle_mask = car_mask | truck_mask | bus_mask
                info['annos']['name'][vehicle_mask] = "vehicle"

                pedestrian_mask = person_mask
                info['annos']['name'][pedestrian_mask] = "pedestrian"

                cyclist_mask = bicycle_mask | motorbike_mask
                info['annos']['name'][cyclist_mask] = "cyclist"

        vehicle_total = car_total + truck_total + bus_total
        pedestrian_total = person_total
        cyclist_total = bicycle_total + motorbike_total
        print("num vehicle: ", vehicle_total)
        print("num pedestrian: ", pedestrian_total)
        print("num cyclist: ", cyclist_total)


    def __len__(self):
        return len(self.infos)

    def __getitem__(self, idx):
        info = self.infos[idx]
        # read input

        v_path = self.data_root / info['velodyne_path']
        points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, self.num_point_features])

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
            if self.augm:
                # sample

                # instance
                augm_class_mask = np.array([n in self.augm_class for n in annos["name"]], dtype=np.bool_)
                agm.noise_per_object(gt_boxes, points, augm_class_mask)

                # global
                gt_boxes, points = agm.random_flip(gt_boxes, points)
                gt_boxes, points = agm.global_rotation_v2(gt_boxes, points)
                gt_boxes, points = agm.global_scaling_v2(gt_boxes, points, min_scale=0.95, max_scale=1.05)
                gt_boxes, points = agm.global_translate(gt_boxes, points, noise_translate_std=[0.25, 0.25, 0.25])

            # filter range
            bv_range = self.detection_range[[0, 1, 3, 4]]
            range_mask = box_np_ops.filter_gt_box_outside_range(gt_boxes, bv_range)
            gt_boxes = gt_boxes[range_mask]
            gt_classes = gt_classes[range_mask]
            gt_names = gt_names[range_mask]
            difficulty = difficulty[range_mask]

            gt_boxes[:, 6] = box_np_ops.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)
            example['annos'] = {'gt_classes': gt_classes, 'gt_boxes': gt_boxes, 'difficulty': difficulty,
                                'gt_names': gt_names}
            np.random.shuffle(points)

        self.points = points

        voxels, coors, num_points_per_voxel = self.voxel_generator.generate(points)

        grid_size = self.grid_size
        voxel_size = self.voxel_generator.voxel_size
        offset = self.voxel_generator.offset
        anchors_mask = self.anchor_assigner.create_mask(coors, grid_size, voxel_size, offset, gpu=self.create_mask_gpu)

        example['voxels'] = voxels
        example['coordinates'] = coors
        example['num_points_per_voxel'] = num_points_per_voxel
        example['anchors_mask'] = anchors_mask
        example['points'] = points

        if self.training:
            gt_classes = example['annos']['gt_classes']
            gt_boxes = example['annos']['gt_boxes']
            label, bbox_targets, bbox_outside_weights, dir_targets = self.anchor_assigner.assign(gt_classes,
                                                                                                 gt_boxes,
                                                                                                 anchors_mask)

            example['labels'] = label
            example['bbox_targets'] = bbox_targets
            example['dir_targets'] = dir_targets
            example['bbox_outside_weights'] = bbox_outside_weights

        return example

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.infos):
            info = self.infos[self.n]
            v_path = self.data_root / info['velodyne_path']
            points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, self.num_point_features])
            voxels, coors, num_points_per_voxel = self.voxel_generator.generate(points)
            grid_size = self.grid_size
            voxel_size = self.voxel_generator.voxel_size
            offset = self.voxel_generator.offset
            anchors_mask = self.anchor_assigner.create_mask(coors, grid_size, voxel_size, offset)
            example = {'voxels': voxels, 'coordinates': coors, 'num_points_per_voxel': num_points_per_voxel,
                       'anchors_mask': anchors_mask}
            self.n += 1
            return example
        else:
            raise StopIteration


class InferData:

    def __init__(self, config, voxel_generator, anchor_assigner, dtype=torch.float32):
        self.voxel_generator = voxel_generator
        self.anchor_assigner = anchor_assigner
        self.grid_size = config['grid_size']
        self.create_mask_gpu = config['create_mask_gpu'] == 1
        self.dtype = dtype

        self.voxel_time = 0.0
        self.mask_time = 0.0
        self.convert_time = 0.0
        self.device = config['device']

    def get(self, points):
        start = time.time()
        voxels, coors, num_points_per_voxel = self.voxel_generator.generate(points)
        voxel_time = time.time()
        anchors_mask = self.anchor_assigner.create_mask(coors, self.grid_size, self.voxel_generator.voxel_size,
                                                        self.voxel_generator.offset, gpu=self.create_mask_gpu)
        anchors_mask = anchors_mask[np.newaxis, :]
        mask_time = time.time()
        example = {'voxels': voxels, 'coordinates': coors, 'num_points_per_voxel': num_points_per_voxel,
                   'anchors_mask': anchors_mask}
        example = example_convert_to_torch(example, self.dtype, device=self.device)
        convert_time = time.time()

        self.voxel_time += voxel_time - start
        self.mask_time += mask_time - voxel_time
        self.convert_time += convert_time - mask_time
        return example
