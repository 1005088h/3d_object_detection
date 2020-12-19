import framework.box_torch_ops as box_torch_ops
import time
import numpy as np
import framework.box_np_ops as box_np_ops

from numba import cuda

class AnchorAssigner:
    def __init__(self, config):
        config['detect_class'] = ["vehicle"] # vehicle, pedestrian, cyclist
        # config['detect_class'] = ["vehicle", "pedestrian", "cyclist"]  # vehicle, pedestrian, cyclist
        self.detect_class = config['detect_class']
        config["vehicle"] = {}
        config["vehicle"]["sizes"] = [[4.6, 2.10, 1.8], [7.5, 2.6, 2.9], [12.6, 2.9, 3.8]]
        # config["vehicle"]["sizes"] = [[4.6, 2.10, 1.8]]
        config["vehicle"]["rotations"] = [0, 1.5707963267948966]
        config["vehicle"]["matched_threshold"] = 0.6
        config["vehicle"]["unmatched_threshold"] = 0.45

        config["pedestrian"] = {}
        config["pedestrian"]["sizes"] = [[0.96874749, 0.9645992, 1.81212425]]
        config["pedestrian"]["rotations"] = [0]
        config["pedestrian"]["matched_threshold"] = 0.45
        config["pedestrian"]["unmatched_threshold"] = 0.25

        config["cyclist"] = {}
        config["cyclist"]["sizes"] = [[2.02032733, 0.98075615, 1.72027404]]
        config["cyclist"]["rotations"] = [0, 1.5707963267948966]
        config["cyclist"]["matched_threshold"] = 0.5
        config["cyclist"]["unmatched_threshold"] = 0.25

        self.feature_map_size = np.array(config['feature_map_size'], dtype=np.float32)
        self.anchor_strides = config['detection_range_diff'] / self.feature_map_size
        self.anchor_offsets = config['detection_offset']
        self.grid_size = config['grid_size']
        self.box_code_size = config['box_code_size']

        self.anchors = []
        self.anchors_bv = []
        self.matched_threshold = []
        self.unmatched_threshold = []
        self.class_masks = {}
        #self.names = []
        #self.class_anchor = {}
        start_index = 0

        for cls in self.detect_class:
            self.sizes = config[cls]["sizes"]
            self.rotations = config[cls]["rotations"]
            matched_threshold = config[cls]['matched_threshold']
            unmatched_threshold = config[cls]['unmatched_threshold']

            anchors = self.generate().reshape([-1, 7])
            anchors_bv = box_np_ops.rbbox2d_to_near_bbox(anchors[:, [0, 1, 3, 4, 6]])
            num_anchors = anchors.shape[0]
            matched_threshold = np.full(num_anchors, matched_threshold, anchors.dtype)
            unmatched_threshold = np.full(num_anchors, unmatched_threshold, anchors.dtype)
            #name = np.full(num_anchors, cls)

            self.anchors.append(anchors)
            self.anchors_bv.append(anchors_bv)
            self.matched_threshold.append(matched_threshold)
            self.unmatched_threshold.append(unmatched_threshold)
            #self.names.append(name)

            end_index = start_index + num_anchors
            self.class_masks[cls] = [start_index, end_index]
            start_index = end_index

        self.anchors = np.concatenate(self.anchors)
        self.anchors_bv = np.concatenate(self.anchors_bv)
        self.matched_threshold = np.concatenate(self.matched_threshold)
        self.unmatched_threshold = np.concatenate(self.unmatched_threshold)
        #self.names = np.concatenate(self.names)

        voxel_size = np.array(config['voxel_size'], dtype=np.float32)
        offset = np.array(config['detection_offset'], dtype=np.float32)
        self.anchors_coors = box_np_ops.get_anchor_coor(self.anchors_bv, voxel_size, offset, self.grid_size)
        self.anchors_coors_cuda = cuda.to_device(self.anchors_coors)
        anchors_mask = np.zeros(self.anchors_bv.shape[0], dtype=np.bool)
        self.anchors_mask_cuda = cuda.to_device(anchors_mask)

    def generate(self):
        x_stride, y_stride, z_stride = self.anchor_strides
        x_offset, y_offset, z_offset = self.anchor_offsets + self.anchor_strides / 2
        z_offset = 0
        for s in self.sizes:
            z_offset += s[2]
        z_offset = z_offset / len(self.sizes)
        x_centers = np.arange(self.feature_map_size[0], dtype=np.float32)
        y_centers = np.arange(self.feature_map_size[1], dtype=np.float32)
        z_centers = np.arange(self.feature_map_size[2], dtype=np.float32)

        x_centers = x_centers * x_stride + x_offset
        y_centers = y_centers * y_stride + y_offset
        z_centers = z_centers * z_stride + z_offset

        rotations = np.array(self.rotations, dtype=np.float32)
        sizes = np.reshape(np.array(self.sizes, dtype=np.float32), [-1, 3])
        rets = np.meshgrid(x_centers, y_centers, z_centers, rotations, indexing='ij')
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
            rets[i] = rets[i][..., np.newaxis]
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
        sizes = np.tile(sizes, tile_size_shape)
        rets.insert(3, sizes)
        ret = np.concatenate(rets, axis=-1)
        return ret

    def create_mask(self, coors, grid_size, voxel_size, offset):
        '''
        ### CPU
        anchors_bv = self.anchors_bv
        dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(coors, tuple(grid_size[:-1])) # dense_voxel_map = box_np_ops.cumsum_gpu(dense_voxel_map)
        dense_voxel_map = dense_voxel_map.cumsum(0)
        dense_voxel_map = dense_voxel_map.cumsum(1)
        anchors_area = box_np_ops.fused_get_anchors_area(dense_voxel_map, anchors_bv, voxel_size, offset, grid_size)
        anchors_mask_cpu = anchors_area > 0
        '''
        ### CPU
        anchors_mask = box_np_ops.fused_get_anchors_mask_gpu(coors, tuple(grid_size[:-1]), self.anchors_coors_cuda, self.anchors_mask_cuda)
        '''
        comparison = anchors_mask == anchors_mask_cpu
        identical = comparison.all()
        print(identical)
        '''
        return anchors_mask

    def assign(self, gt_classes_all, gt_boxes_all, anchors_mask_all):
        label_list = []
        bbox_targets_list = []
        bbox_outside_weights_list = []
        dir_cls_targets_list = []
        for cls, index in self.class_masks.items():
            current_class = self.detect_class.index(cls) + 1
            mask = gt_classes_all == current_class
            #gt_classes = gt_classes_all[mask]
            gt_boxes = gt_boxes_all[mask]
            anchors = self.anchors[index[0]: index[1]]
            anchors_mask = anchors_mask_all[index[0]: index[1]]
            matched_threshold = self.matched_threshold[index[0]: index[1]]
            unmatched_threshold = self.unmatched_threshold[index[0]: index[1]]
            num_anchors = anchors.shape[0]

            inds_inside = np.where(anchors_mask)[0]
            anchors = anchors[inds_inside, :]
            matched_threshold = matched_threshold[inds_inside]
            unmatched_threshold = unmatched_threshold[inds_inside]
            num_inside = len(inds_inside)

            labels = -np.ones((num_inside,), dtype=np.int32)  # bg: 0, pos: >0
            bbox_targets = np.zeros((num_inside, self.box_code_size), dtype=self.anchors.dtype)

            if len(gt_boxes) > 0 and anchors.shape[0] > 0:
                # Compute overlaps between the anchors and the gt boxes overlaps
                anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
                # Map from anchor to gt box that has highest overlap
                anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
                # For each anchor, amount of overlap with most overlapping gt box
                anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside), anchor_to_gt_argmax]
                # Map from gt box to an anchor that has highest overlap
                gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
                # For each gt box, amount of overlap with most overlapping anchor
                gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]
                # must remove gt which doesn't match any anchor.
                empty_gt_mask = gt_to_anchor_max == 0
                gt_to_anchor_max[empty_gt_mask] = -1
                # Find all anchors that share the max overlap amount
                # (this includes many ties)
                anchors_with_max_overlap = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]
                # Fg label: for each gt use anchors with highest overlap
                # (including ties)
                gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
                labels[anchors_with_max_overlap] = 1#gt_classes[gt_inds_force]

                # Fg label: above threshold IOU
                pos_inds = anchor_to_gt_max >= matched_threshold
                gt_inds = anchor_to_gt_argmax[pos_inds]
                labels[pos_inds] = 1 #gt_classes[gt_inds]

                # Bg label: below threshold IOU
                bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
                labels[bg_inds] = 0

                # Re-assign max overlap gt if all below threshold IOU
                labels[anchors_with_max_overlap] = 1#gt_classes[gt_inds_force]

                fg_inds = np.where(labels > 0)[0]
                bbox_targets[fg_inds, :] = box_np_ops.box_encode(gt_boxes[anchor_to_gt_argmax[fg_inds], :],
                                                                 anchors[fg_inds, :])
            else:
                labels[:] = 0

            #self.class_anchor[cls] += np.sum(labels > 0)

            bbox_outside_weights = np.zeros((num_inside,), dtype=self.anchors.dtype)
            bbox_outside_weights[labels > 0] = 1.0

            dir_cls_targets = None
            # Map up to original set of anchors
            if inds_inside is not None:
                labels = unmap(labels, num_anchors, inds_inside, fill=-1)
                bbox_targets = unmap(bbox_targets, num_anchors, inds_inside, fill=0)
                bbox_outside_weights = unmap(bbox_outside_weights, num_anchors, inds_inside, fill=0)
                dir_cls_targets = get_direction_target(self.anchors[index[0]: index[1]], bbox_targets)

            label_list.append(labels)
            bbox_targets_list.append(bbox_targets)
            bbox_outside_weights_list.append(bbox_outside_weights)
            dir_cls_targets_list.append(dir_cls_targets)

        labels = np.concatenate(label_list)
        bbox_targets = np.concatenate(bbox_targets_list)
        bbox_outside_weights = np.concatenate(bbox_outside_weights_list)
        dir_cls_targets = np.concatenate(dir_cls_targets_list)

        return labels, bbox_targets, bbox_outside_weights, dir_cls_targets


def similarity_fn(anchors, gt_boxes):
    anchors_rbv = anchors[:, [0, 1, 3, 4, 6]]
    gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
    boxes1_bv = box_np_ops.rbbox2d_to_near_bbox(anchors_rbv)
    boxes2_bv = box_np_ops.rbbox2d_to_near_bbox(gt_boxes_rbv)
    ret = box_np_ops.iou_jit(boxes1_bv, boxes2_bv, eps=0.0)
    return ret


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def get_direction_target(anchors, reg_targets):
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = rot_gt > 0
    return dir_cls_targets.astype('int32')


'''
inds_inside = np.where(anchors_mask)[0]
anchors = self.anchors[inds_inside, :]
matched_threshold = self.matched_threshold[inds_inside]
unmatched_threshold = self.unmatched_threshold[inds_inside]
num_inside = len(inds_inside)

labels = -np.ones((num_inside,), dtype=np.int32)
bbox_targets = np.zeros((num_inside, self.box_code_size), dtype=self.anchors.dtype)

if len(gt_boxes) > 0 and anchors.shape[0] > 0:
    # Compute overlaps between the anchors and the gt boxes overlaps
    anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
    # Map from anchor to gt box that has highest overlap
    anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    # For each anchor, amount of overlap with most overlapping gt box
    anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_inside), anchor_to_gt_argmax]
    # Map from gt box to an anchor that has highest overlap
    gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
    # For each gt box, amount of overlap with most overlapping anchor
    gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]
    # must remove gt which doesn't match any anchor.
    empty_gt_mask = gt_to_anchor_max == 0
    gt_to_anchor_max[empty_gt_mask] = -1
    # Find all anchors that share the max overlap amount
    # (this includes many ties)
    anchors_with_max_overlap = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]
    # Fg label: for each gt use anchors with highest overlap
    # (including ties)
    gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
    labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

    # Fg label: above threshold IOU
    pos_inds = anchor_to_gt_max >= matched_threshold
    gt_inds = anchor_to_gt_argmax[pos_inds]
    labels[pos_inds] = gt_classes[gt_inds]

    # Bg label: below threshold IOU
    bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    labels[bg_inds] = 0

    # Re-assign max overlap gt if all below threshold IOU
    labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]

    fg_inds = np.where(labels > 0)[0]
    bbox_targets[fg_inds, :] = box_np_ops.box_encode(gt_boxes[anchor_to_gt_argmax[fg_inds], :],
                                                     anchors[fg_inds, :])
else:
    labels[:] = 0

bbox_outside_weights = np.zeros((num_inside,), dtype=self.anchors.dtype)
bbox_outside_weights[labels > 0] = 1.0

dir_cls_targets = None
# Map up to original set of anchors
if inds_inside is not None:
    labels = unmap(labels, self.num_anchors, inds_inside, fill=-1)
    bbox_targets = unmap(bbox_targets, self.num_anchors, inds_inside, fill=0)
    bbox_outside_weights = unmap(bbox_outside_weights, self.num_anchors, inds_inside, fill=0)
    dir_cls_targets = get_direction_target(self.anchors, bbox_targets)

return labels, bbox_targets, bbox_outside_weights, dir_cls_targets
'''
