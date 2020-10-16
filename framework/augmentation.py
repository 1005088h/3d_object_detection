import numba
import numpy as np
#from second.core.geometry import (points_in_convex_polygon_3d_jit,
#                                    points_in_convex_polygon_jit)
from framework import box_np_ops
from framework.box_np_ops import points_in_convex_polygon_3d_jit


def random_flip(gt_boxes, points):
    enable = np.random.random() > 0.5
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def global_rotation(gt_boxes, points, rotation=np.pi / 4):
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points

def global_rotation_v2(gt_boxes, points):
    pitch = 2
    pitch = np.random.uniform(-pitch, pitch)
    pitch = pitch / 180 * np.pi
    points[:, :3] = box_np_ops.rotation_points_single_angle(points[:, :3], pitch, axis=1)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(gt_boxes[:, :3], pitch, axis=1)

    roll = 1
    roll = np.random.uniform(-roll, roll)
    roll = roll / 180 * np.pi
    points[:, :3] = box_np_ops.rotation_points_single_angle(points[:, :3], roll, axis=0)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(gt_boxes[:, :3], roll, axis=0)

    yaw = 40
    yaw = np.random.uniform(-yaw, yaw)
    yaw = yaw / 180 * np.pi
    points[:, :3] = box_np_ops.rotation_points_single_angle(points[:, :3], yaw, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(gt_boxes[:, :3], yaw, axis=2)
    gt_boxes[:, 6] += yaw
    return gt_boxes, points


def global_scaling(gt_boxes, points, min_scale=0.95, max_scale=1.05):
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points

def global_scaling_v2(gt_boxes, points, min_scale=0.85, max_scale=1.15):
    x_scale = np.random.uniform(0.85, 1.15)
    y_scale = np.random.uniform(0.85, 1.15)
    z_scale = np.random.uniform(0.95, 1.05)
    scales = np.array([x_scale, y_scale, z_scale])
    points[:, :3] *= scales
    gt_boxes[:, :3] *= scales

    gt_boxes[:, 3] *= np.sqrt(np.square(x_scale * np.cos(gt_boxes[:, 6])) + np.square(y_scale * np.sin(gt_boxes[:, 6])))
    gt_boxes[:, 4] *= np.sqrt(np.square(x_scale * np.sin(gt_boxes[:, 6])) + np.square(y_scale * np.cos(gt_boxes[:, 6])))
    gt_boxes[:, 5] *= z_scale
    r = np.tan(gt_boxes[:, 6])
    r = r * (y_scale / x_scale)
    gt_boxes[:, 6] = np.arctan(r)
    return gt_boxes, points


def global_translate(gt_boxes, points, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    noise_translate = np.array([np.random.normal(0, noise_translate_std[0], 1),
                                np.random.normal(0, noise_translate_std[1], 1),
                                np.random.normal(0, noise_translate_std[2], 1)]).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points


def noise_per_object(gt_boxes,
                     points=None,
                     rotation_perturb=np.pi / 4,
                     center_noise_std=1.0,
                     global_random_rot_range=np.pi / 4,
                     num_try=100):

    num_boxes = gt_boxes.shape[0]
    rotation_perturb = [-rotation_perturb, rotation_perturb]
    global_random_rot_range = [-global_random_rot_range, global_random_rot_range]
    enable_grot = np.abs(global_random_rot_range[0] - global_random_rot_range[1]) >= 1e-3
    center_noise_std = [center_noise_std, center_noise_std, center_noise_std]
    valid_mask = np.ones((num_boxes, ), dtype=np.bool_)

    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(scale=center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=[0.5, 0.5, 0.5],
        axis=2)

    if not enable_grot:
        selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                       valid_mask, loc_noises, rot_noises)
    else:
        selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises,
                                           rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                      rot_transforms, valid_mask)
    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)

def remove_points_in_boxes(points, boxes):
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[np.logical_not(masks.any(-1))]
    return points


def remove_points_outside_boxes(points, boxes):
    masks = box_np_ops.points_in_rbbox(points, boxes)
    points = points[masks.any(-1)]
    return points


@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:#pitch
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:#yaw
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:#roll
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@numba.jit(nopython=True)
def _box_single_to_corner_jit(boxes):
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(
        1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask


@numba.njit
def noise_per_box_group(boxes, valid_mask, loc_noises, rot_noises, group_nums):
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_groups = group_nums.shape[0]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_corners[i] = box_corners[i + idx]
                    current_corners[i] -= boxes[i + idx, :2]
                    _rotation_box2d_jit_(current_corners[i],
                                         rot_noises[idx + i, j], rot_mat_T)
                    current_corners[
                        i] += boxes[i + idx, :2] + loc_noises[i + idx, j, :2]
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners)
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_group_v2_(boxes, valid_mask, loc_noises, rot_noises,
                            group_nums, global_rot_noises):
    # WARNING: this function need boxes to be sorted by group id.
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    current_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((max_group_num, 2), dtype=boxes.dtype)

    current_grot = np.zeros((max_group_num, ), dtype=boxes.dtype)
    dst_grot = np.zeros((max_group_num, ), dtype=boxes.dtype)

    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)

    # print(valid_mask)
    idx = 0
    for num in group_nums:
        if valid_mask[idx]:
            for j in range(num_tests):
                for i in range(num):
                    current_box[0, :] = boxes[i + idx]
                    current_radius = np.sqrt(current_box[0, 0]**2 +
                                             current_box[0, 1]**2)
                    current_grot[i] = np.arctan2(current_box[0, 0],
                                                 current_box[0, 1])
                    dst_grot[
                        i] = current_grot[i] + global_rot_noises[idx + i, j]
                    dst_pos[i, 0] = current_radius * np.sin(dst_grot[i])
                    dst_pos[i, 1] = current_radius * np.cos(dst_grot[i])
                    current_box[0, :2] = dst_pos[i]
                    current_box[0, -1] += (dst_grot[i] - current_grot[i])

                    rot_sin = np.sin(current_box[0, -1])
                    rot_cos = np.cos(current_box[0, -1])
                    rot_mat_T[0, 0] = rot_cos
                    rot_mat_T[0, 1] = -rot_sin
                    rot_mat_T[1, 0] = rot_sin
                    rot_mat_T[1, 1] = rot_cos
                    current_corners[
                        i] = current_box[0, 2:
                                         4] * corners_norm @ rot_mat_T + current_box[0, :
                                                                                     2]
                    current_corners[i] -= current_box[0, :2]

                    _rotation_box2d_jit_(current_corners[i],
                                         rot_noises[idx + i, j], rot_mat_T)
                    current_corners[
                        i] += current_box[0, :2] + loc_noises[i + idx, j, :2]
                coll_mat = box_collision_test(
                    current_corners[:num].reshape(num, 4, 2), box_corners)
                for i in range(num):  # remove self-coll
                    coll_mat[i, idx:idx + num] = False
                if not coll_mat.any():
                    for i in range(num):
                        success_mask[i + idx] = j
                        box_corners[i + idx] = current_corners[i]
                        loc_noises[i + idx, j, :2] += (
                            dst_pos[i] - boxes[i + idx, :2])
                        rot_noises[i + idx, j] += (
                            dst_grot[i] - current_grot[i])
                    break
        idx += num
    return success_mask


@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises,
                      global_rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2, ), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0]**2 + boxes[i, 1]**2)
                current_grot = np.arctan2(boxes[i, 1], boxes[i, 0])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.cos(dst_grot)
                dst_pos[1] = current_radius * np.sin(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = current_box[0, 2:4] * corners_norm @ rot_mat_T + current_box[0, :2]
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j], rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break
    return success_mask


@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, valid_mask):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(transform, indices):
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@numba.njit
def group_transform_(loc_noise, rot_noise, locs, rots, group_center,
                     valid_mask):
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j]) - np.sin(rot_center))
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j]) - np.cos(rot_center))


@numba.njit
def group_transform_v2_(loc_noise, rot_noise, locs, rots, group_center,
                        grot_noise, valid_mask):
    # loc_noise: [N, M, 3], locs: [N, 3]
    # rot_noise: [N, M]
    # group_center: [N, 3]
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j] + grot_noise[i, j]) -
                    np.sin(rot_center + grot_noise[i, j]))
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j] + grot_noise[i, j]) -
                    np.cos(rot_center + grot_noise[i, j]))


def set_group_noise_same_(loc_noise, rot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]


def set_group_noise_same_v2_(loc_noise, rot_noise, grot_noise, group_ids):
    gid_to_index_dict = {}
    for i, gid in enumerate(group_ids):
        if gid not in gid_to_index_dict:
            gid_to_index_dict[gid] = i
    for i in range(loc_noise.shape[0]):
        loc_noise[i] = loc_noise[gid_to_index_dict[group_ids[i]]]
        rot_noise[i] = rot_noise[gid_to_index_dict[group_ids[i]]]
        grot_noise[i] = grot_noise[gid_to_index_dict[group_ids[i]]]


def get_group_center(locs, group_ids):
    num_groups = 0
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    group_id_dict = {}
    group_id_num_dict = OrderedDict()
    for i, gid in enumerate(group_ids):
        if gid >= 0:
            if gid in group_id_dict:
                group_centers[group_id_dict[gid]] += locs[i]
                group_id_num_dict[gid] += 1
            else:
                group_id_dict[gid] = num_groups
                num_groups += 1
                group_id_num_dict[gid] = 1
                group_centers[group_id_dict[gid]] = locs[i]
    for i, gid in enumerate(group_ids):
        group_centers_ret[
            i] = group_centers[group_id_dict[gid]] / group_id_num_dict[gid]
    return group_centers_ret, group_id_num_dict




def noise_per_object_v2_(gt_boxes,
                         points=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=np.pi / 4,
                         num_try=100):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]

    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes, ), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])

    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2)
    if np.abs(global_random_rot_range[0] - global_random_rot_range[1]) < 1e-3:
        selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                       valid_mask, loc_noises, rot_noises)
    else:
        selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises,
                                           global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    if points is not None:
        surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)




'''
def global_rotation_v2(gt_boxes, points, min_rad=-np.pi / 4,
                       max_rad=np.pi / 4):
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2)
    gt_boxes[:, 6] += noise_rotation
    return gt_boxes, points
'''


@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret




'''
def noise_per_object_v3_(gt_boxes,
                         points=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=np.pi / 4,
                         num_try=100,
                         group_ids=None):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]
    enable_grot = np.abs(global_random_rot_range[0] -
                         global_random_rot_range[1]) >= 1e-3
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes, ), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    gt_grots = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lowers = global_random_rot_range[0] - gt_grots
    grot_uppers = global_random_rot_range[1] - gt_grots
    global_rot_noises = np.random.uniform(
        grot_lowers[..., np.newaxis],
        grot_uppers[..., np.newaxis],
        size=[num_boxes, num_try])
    if group_ids is not None:
        if enable_grot:
            set_group_noise_same_v2_(loc_noises, rot_noises, global_rot_noises,
                                     group_ids)
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
        group_centers, group_id_num_dict = get_group_center(
            gt_boxes[:, :3], group_ids)
        if enable_grot:
            group_transform_v2_(loc_noises, rot_noises, gt_boxes[:, :3],
                                gt_boxes[:, 6], group_centers,
                                global_rot_noises, valid_mask)
        else:
            group_transform_(loc_noises, rot_noises, gt_boxes[:, :3],
                             gt_boxes[:, 6], group_centers, valid_mask)
        group_nums = np.array(list(group_id_num_dict.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0]
    gt_box_corners = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2)
    if group_ids is not None:
        if not enable_grot:
            selected_noise = noise_per_box_group(gt_boxes[:, [0, 1, 3, 4, 6]],
                                                 valid_mask, loc_noises,
                                                 rot_noises, group_nums)
        else:
            selected_noise = noise_per_box_group_v2_(
                gt_boxes[:, [0, 1, 3, 4, 6]], valid_mask, loc_noises,
                rot_noises, group_nums, global_rot_noises)
    else:
        if not enable_grot:
            selected_noise = noise_per_box(gt_boxes[:, [0, 1, 3, 4, 6]],
                                           valid_mask, loc_noises, rot_noises)
        else:
            selected_noise = noise_per_box_v2_(gt_boxes[:, [0, 1, 3, 4, 6]],
                                               valid_mask, loc_noises,
                                               rot_noises, global_rot_noises)
    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
        points_transform_(points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, valid_mask)

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)
'''