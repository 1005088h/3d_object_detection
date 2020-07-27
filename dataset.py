from torch.utils.data import Dataset
import pickle
import numpy as np
import box_np_ops


class GenericDataset(Dataset):
    def __init__(self, config, train_info_path, voxel_generator, anchor_assigner, training=True):
        with open(train_info_path, 'rb') as f:
            self._infos = pickle.load(f)

        self._num_point_features = config['num_point_features']
        self.voxel_generator = voxel_generator
        self.anchor_assigner = anchor_assigner
        self._detect_class = config['detect_class']
        self._detection_range = config['detection_range']
        self._grid_size = config['grid_size']
        self.training = training

        for info in self._infos:
            difficulty_mask = info['annos']["num_points"] > 0
            for key in info['annos']:
                info['annos'][key] = info['annos'][key][difficulty_mask]

    def __len__(self):
        return len(self._infos)

    def __getitem__(self, idx):
        info = self._infos[idx]
        # read input
        points = np.fromfile(str(info['velodyne_path']), dtype=np.float32, count=-1).reshape(
            [-1, self._num_point_features])

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
        if info['annos'] is not None:
            annos = info['annos']
            # filter class
            gt_class_mask = np.array([n in self._detect_class for n in annos["name"]], dtype=np.bool_)
            gt_names = annos["name"][gt_class_mask]
            gt_classes = np.array([self._detect_class.index(n) + 1 for n in gt_names], dtype=np.int32)
            loc = annos["location"][gt_class_mask]
            dims = annos["dimensions"][gt_class_mask]
            rots = annos["rotation_y"][gt_class_mask]
            difficulty = annos["difficulty"][gt_class_mask]
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            # data augmentation
            '''
            gt_boxes, points = prep.random_flip(gt_boxes, points)
            gt_boxes, points = prep.global_rotation(gt_boxes, points, rotation=global_rotation_noise)
            gt_boxes, points = prep.global_scaling_v2(gt_boxes, points, *global_scaling_noise)
            gt_boxes, points = prep.global_translate(gt_boxes, points, global_loc_noise_std)
            '''

            # filter range
            bv_range = self._detection_range[[0, 1, 3, 4]]
            range_mask = box_np_ops.filter_gt_box_outside_range(gt_boxes, bv_range)
            gt_boxes = gt_boxes[range_mask]
            gt_classes = gt_classes[range_mask]
            difficulty = difficulty[range_mask]
            gt_boxes[:, 6] = box_np_ops.limit_period(gt_boxes[:, 6], offset=0.5, period=2 * np.pi)

            example['annos'] = {'gt_classes': gt_classes, 'gt_boxes': gt_boxes, 'difficulty': difficulty}

            # if self.training:
            np.random.shuffle(points)

        voxels, coors, num_points_per_voxel = self.voxel_generator.generate(points)
        grid_size = self._grid_size
        voxel_size = self.voxel_generator.voxel_size
        offset = self.voxel_generator.offset
        anchors_mask = self.anchor_assigner.create_mask(coors, grid_size, voxel_size, offset)

        example['voxels'] = voxels
        example['coordinates'] = coors
        example['num_points_per_voxel'] = num_points_per_voxel
        example['anchors_mask'] = anchors_mask

        if self.training:
            gt_classes = example['annos']['gt_classes']
            gt_boxes = example['annos']['gt_boxes']
            label, bbox_targets, bbox_outside_weights, dir_cls_targets = self.anchor_assigner.assign(gt_classes, gt_boxes, anchors_mask)
            example['labels'] = label
            example['bbox_targets'] = bbox_targets
            example['bbox_outside_weights'] = bbox_outside_weights
            example['dir_cls_targets'] = dir_cls_targets

        return example
