import framework.box_np_ops as box_np_ops
import numpy as np
import framework.box_torch_ops as box_torch_ops
import torch
from framework.nms import nms_gpu
import time


class Inference:
    def __init__(self, config, anchor_assigner):
        self.anchors = torch.from_numpy(anchor_assigner.anchors).cuda()
        self.names = anchor_assigner.names
        self._nms_pre_max_size = 1000
        self._nms_post_max_size = 300
        self._nms_iou_threshold = 0.1
        self._box_code_size = 7
        self._num_class = 1
        self._use_direction_classifier = True
        self._nms_score_threshold = 0.05
        self.center_limit = config['center_limit']
        self.detect_class = np.array(config['detect_class'])


    def infer(self, example, preds_dict):
        annos = []
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_size = batch_box_preds.shape[0]

        batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_code_size)
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, self._num_class)

        batch_anchors = torch.stack([self.anchors] * batch_size)
        batch_box_preds = box_torch_ops.box_decode(batch_box_preds, batch_anchors)

        batch_anchors_mask = example["anchors_mask"]
        batch_dir_preds = preds_dict["dir_cls_preds"]
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        image_idx = example["image_idx"]

        for box_preds, cls_preds, dir_preds, a_mask, img_id in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_anchors_mask, image_idx
        ):
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
            name_preds = self.names[a_mask.cpu()]

            cls_scores = torch.sigmoid(cls_preds)
            #top_scores, top_labels = torch.max(cls_scores, dim=-1)
            top_scores = torch.max(cls_scores, dim=-1)[0]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            thresh = torch.tensor([self._nms_score_threshold], device=top_scores.device).type_as(top_scores)
            selected = None
            top_scores_keep = top_scores >= thresh
            if top_scores_keep.any():
                top_scores = top_scores[top_scores_keep]
                box_preds = box_preds[top_scores_keep]
                name_preds = name_preds[top_scores_keep.cpu()]

                #if self._use_direction_classifier:
                dir_labels = dir_labels[top_scores_keep]
                #top_labels = top_labels[top_scores_keep]

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)

                selected = nms(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                )

            anno = get_start_result_anno()
            if selected is not None:
                box_preds = box_preds[selected]
                #labels_preds = top_labels[selected]
                scores_preds = top_scores[selected]
                name_preds = name_preds[selected.cpu()]
                if self._use_direction_classifier:
                    dir_preds = dir_labels[selected].bool()
                    dir_mask = (box_preds[..., -1] > 0)
                    opp_labels = dir_mask ^ dir_preds
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))

                scores_preds = scores_preds.detach().cpu().numpy()
                box_preds = box_preds.detach().cpu().numpy()
                #label_preds = labels_preds.detach().cpu().numpy()

                limit_range = self.center_limit
                min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask

                box_preds = box_preds[range_mask]
                box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                #label_preds = label_preds[range_mask]
                scores_preds = scores_preds[range_mask]
                name_preds = name_preds[range_mask]

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    anno["name"] = name_preds#self.detect_class[label_preds]
                    anno["location"] = box_preds[:, :3]
                    anno["dimensions"] = box_preds[:, 3:6]
                    anno["rotation_y"] = box_preds[:, 6]
                    anno["score"] = scores_preds
                anno["image_idx"] = np.array([img_id] * dt_num)
            annos.append(anno)
        return annos


def get_start_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations


def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    dets = torch.cat([bboxes, scores.unsqueeze(-1).float()], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().cuda()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().cuda()
