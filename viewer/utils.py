import json
from pathlib import Path
from framework import box_np_ops
import numpy as np
from shapely.geometry import Polygon
class Settings:
    def __init__(self, cfg_path):

        self._cfg_path = cfg_path
        self._settings = {}
        self._setting_defaultvalue = {}
        if not Path(self._cfg_path).exists():
            with open(self._cfg_path, 'w') as f:
                f.write(json.dumps(self._settings, indent=2, sort_keys=True))
        else:
            with open(self._cfg_path, 'r') as f:
                self._settings = json.loads(f.read())

    def set(self, name, value):
        self._settings[name] = value
        with open(self._cfg_path, 'w') as f:
            f.write(json.dumps(self._settings, indent=2, sort_keys=True))

    def get(self, name, default_value=None):
        if name in self._settings:
            return self._settings[name]
        if default_value is None:
            raise ValueError("name not exist")
        return default_value

    def save(self, path):
        with open(path, 'w') as f:
            f.write(json.dumps(self._settings, indent=2, sort_keys=True))

    def load(self, path):
        with open(self._cfg_path, 'r') as f:
            self._settings = json.loads(f.read())
            
 
def riou3d_shapely(rbboxes1, rbboxes2):
    N, K = rbboxes1.shape[0], rbboxes2.shape[0]
    corners1 = box_np_ops.center_to_corner_box2d(
        rbboxes1[:, :2], rbboxes1[:, 3:5], rbboxes1[:, 6])
    corners2 = box_np_ops.center_to_corner_box2d(
        rbboxes2[:, :2], rbboxes2[:, 3:5], rbboxes2[:, 6])
    iou = np.zeros([N, K], dtype=np.float32)
    for i in range(N):
        for j in range(K):
            iw = (min(rbboxes1[i, 2] + rbboxes1[i, 5],
                      rbboxes2[j, 2] + rbboxes2[j, 5]) - max(
                          rbboxes1[i, 2], rbboxes2[j, 2]))
            if iw > 0:
                p1 = Polygon(corners1[i])
                p2 = Polygon(corners2[j])
                inc = p1.intersection(p2).area * iw
                # inc = p1.intersection(p2).area
                if inc > 0:
                    iou[i, j] = inc / (p1.area * rbboxes1[i, 5] +
                                       p2.area * rbboxes2[j, 5] - inc)
                    # iou[i, j] = inc / (p1.area + p2.area - inc)

    return iou


def kitti_anno_to_corners(info, annos=None):

    rect = info['calib/R0_rect']
    P2 = info['calib/P2']
    Tr_velo_to_cam = info['calib/Tr_velo_to_cam']

    if annos is None:
        annos = info['annos']
    dims = annos['dimensions']
    loc = annos['location']
    rots = annos['rotation_y']
    scores = None
    if 'score' in annos:
        scores = annos['score']
    boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect,
                                                 Tr_velo_to_cam)
    boxes_corners = box_np_ops.center_to_corner_box3d(
        boxes_lidar[:, :3],
        boxes_lidar[:, 3:6],
        boxes_lidar[:, 6],
        origin=[0.5, 0.5, 0],
        axis=2)
    return boxes_corners, scores, boxes_lidar


def remove_low_score(detection_anno, thresh):
    img_filtered_annotations = {}

    relevant_annotation_indices = [
        i for i, s in enumerate(detection_anno['score']) if s >= thresh
    ]
    for key in detection_anno.keys():
        if len(detection_anno[key]) > 0:
            img_filtered_annotations[key] = detection_anno[key][relevant_annotation_indices]
        else:
            img_filtered_annotations = detection_anno
    return img_filtered_annotations