import framework.box_np_ops as box_np_ops
import numpy as np
import framework.box_torch_ops as box_torch_ops
import torch
from framework.nms import nms_gpu
import time


class Inference:
    def __init__(self, config, anchor_assigner):
        self.device = config['device']
        self.anchors = torch.from_numpy(anchor_assigner.anchors).to(self.device)
        self._nms_pre_max_size = 1000
        self._nms_post_max_size = 300
        self._nms_iou_threshold = 0.1
        self._box_code_size = 7
        self._num_class = 1
        self._use_direction_classifier = True
        self._nms_score_threshold = torch.tensor([0.05]).to(self.device)
        self.center_limit = config['center_limit']
        self.detect_class = np.array(config['detect_class'])
        self.class_masks = anchor_assigner.class_masks

        self.p1, self.p2, self.p3, self.p4, self.p5 = 0.0, 0.0, 0.0, 0.0, 0.0

    def infer_gpu(self, example, preds_dict):
        annos = []

        cls_preds_all = preds_dict["cls_preds"].squeeze(0)
        box_preds_all = preds_dict["box_preds"].squeeze(0)
        dir_preds_all = preds_dict["dir_preds"].squeeze(0)
        anchors_mask = example["anchors_mask"].squeeze(0)
        name_list, location_list, dimensions_list, rotation_y_list, score_list = [], [], [], [], []

        for cls, a_range in self.class_masks.items():
            torch.cuda.synchronize()
            start = time.time()
            a_mask = anchors_mask[a_range[0]: a_range[1]]
            box_preds = box_preds_all[a_range[0]: a_range[1]]
            cls_preds = cls_preds_all[a_range[0]: a_range[1]]
            dir_preds = dir_preds_all[a_range[0]: a_range[1]]
            anchors = self.anchors[a_range[0]: a_range[1]]

            ## Anchor mask
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
            anchors = anchors[a_mask]

            cls_scores = torch.sigmoid(cls_preds)
            top_scores = torch.max(cls_scores, dim=-1)[0]
            dir_labels = torch.max(dir_preds, dim=-1)[1]

            selected = None
            ## Score mask
            top_scores_keep = top_scores >= self._nms_score_threshold

            if top_scores_keep.any():
                top_scores = top_scores[top_scores_keep]
                box_preds = box_preds[top_scores_keep]
                dir_labels = dir_labels[top_scores_keep]
                anchors = anchors[top_scores_keep]

                # if self._nms_pre_max_size is not None:
                num_keeped_scores = top_scores.shape[0]
                pre_max_size = min(num_keeped_scores, self._nms_pre_max_size)

                top_scores, indices = torch.topk(top_scores, k=pre_max_size)

                top_scores = top_scores.cpu().numpy()
                box_preds = box_preds[indices].cpu().numpy()
                dir_labels = dir_labels[indices].cpu().numpy()
                anchors = anchors[indices].cpu().numpy()

                torch.cuda.synchronize()
                p1 = time.time()

                box_preds = box_np_ops.box_decode(box_preds, anchors)

                torch.cuda.synchronize()
                p2 = time.time()

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                box_preds_corners = box_np_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                boxes_for_nms = box_np_ops.corner_to_standup_nd(box_preds_corners)

                torch.cuda.synchronize()
                p3 = time.time()

                selected = nms(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                )

            torch.cuda.synchronize()
            p4 = time.time()
            if selected is not None:
                ## NMS mask
                box_preds = box_preds[selected]
                scores_preds = top_scores[selected]
                dir_preds = dir_labels[selected].astype(np.bool)
                dir_mask = (box_preds[..., -1] > 0)
                opp_labels = dir_mask ^ dir_preds
                box_preds[..., -1] += np.where(opp_labels, np.pi, 0.0)

                limit_range = self.center_limit
                min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask

                ## Range mask
                box_preds = box_preds[range_mask]
                box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                scores_preds = scores_preds[range_mask]

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    name_list.append(np.full(dt_num, cls, dtype='<U10'))
                    location_list.append(box_preds[:, :3])
                    dimensions_list.append(box_preds[:, 3:6])
                    rotation_y_list.append(box_preds[:, 6])
                    score_list.append(scores_preds)

            self.p1 += p1 - start
            self.p2 += p2 - p1
            self.p3 += p3 - p2
            self.p4 += p4 - p3

        anno = get_start_result_anno()
        if len(name_list) > 0:
            anno["name"] = np.concatenate(name_list)
            anno["location"] = np.concatenate(location_list)
            anno["dimensions"] = np.concatenate(dimensions_list)
            anno["rotation_y"] = np.concatenate(rotation_y_list)
            anno["score"] = np.concatenate(score_list)

        annos.append(anno)
        return annos

    def infer_torch(self, example, preds_dict):
        annos = []

        cls_preds_all = preds_dict["cls_preds"].squeeze(0)
        box_preds_all = preds_dict["box_preds"].squeeze(0)
        dir_preds_all = preds_dict["dir_preds"].squeeze(0)
        anchors_mask = example["anchors_mask"].squeeze(0)
        name_list, location_list, dimensions_list, rotation_y_list, score_list = [], [], [], [], []

        for cls, a_range in self.class_masks.items():
            torch.cuda.synchronize()
            start = time.time()
            a_mask = anchors_mask[a_range[0]: a_range[1]]
            box_preds = box_preds_all[a_range[0]: a_range[1]]
            cls_preds = cls_preds_all[a_range[0]: a_range[1]]
            dir_preds = dir_preds_all[a_range[0]: a_range[1]]
            anchors = self.anchors[a_range[0]: a_range[1]]

            ## Anchor mask
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
            anchors = anchors[a_mask]

            cls_scores = torch.sigmoid(cls_preds)
            top_scores = torch.max(cls_scores, dim=-1)[0]
            dir_labels = torch.max(dir_preds, dim=-1)[1]

            selected = None
            ## Score mask
            top_scores_keep = top_scores >= self._nms_score_threshold
            if top_scores_keep.any():
                top_scores = top_scores[top_scores_keep]
                box_preds = box_preds[top_scores_keep]
                dir_labels = dir_labels[top_scores_keep]
                anchors = anchors[top_scores_keep]

                num_keeped_scores = top_scores.shape[0]
                pre_max_size = min(num_keeped_scores, self._nms_pre_max_size)

                torch.cuda.synchronize()
                p1 = time.time()
                top_scores, indices = torch.topk(top_scores, k=pre_max_size)


                box_preds = box_preds[indices]
                dir_labels = dir_labels[indices]
                anchors = anchors[indices]

                torch.cuda.synchronize()
                p2 = time.time()
                box_preds = box_torch_ops.box_decode(box_preds, anchors)

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)

                torch.cuda.synchronize()
                p3 = time.time()

                selected = nms_torch(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                ).to(self.device)

            torch.cuda.synchronize()
            p4 = time.time()
            if selected is not None:
                ## NMS mask
                box_preds = box_preds[selected]
                scores_preds = top_scores[selected]
                dir_preds = dir_labels[selected].bool()
                dir_mask = (box_preds[..., -1] > 0)
                opp_labels = dir_mask ^ dir_preds
                box_preds[..., -1] += torch.where(opp_labels,
                                                  torch.tensor(np.pi).type_as(box_preds),
                                                  torch.tensor(0.0).type_as(box_preds))

                scores_preds = scores_preds.detach().cpu().numpy()
                box_preds = box_preds.detach().cpu().numpy()

                limit_range = self.center_limit
                min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask

                ## Range mask
                box_preds = box_preds[range_mask]
                box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                scores_preds = scores_preds[range_mask]

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    name_list.append(np.full(dt_num, cls, dtype='<U10'))
                    location_list.append(box_preds[:, :3])
                    dimensions_list.append(box_preds[:, 3:6])
                    rotation_y_list.append(box_preds[:, 6])
                    score_list.append(scores_preds)
            self.p1 += p1 - start
            self.p2 += p2 - p1
            self.p3 += p3 - p2
            self.p4 += p4 - p3

        anno = get_start_result_anno()
        if len(name_list) > 0:
            anno["name"] = np.concatenate(name_list)
            anno["location"] = np.concatenate(location_list)
            anno["dimensions"] = np.concatenate(dimensions_list)
            anno["rotation_y"] = np.concatenate(rotation_y_list)
            anno["score"] = np.concatenate(score_list)

        annos.append(anno)

        return annos

    def infer_v3(self, example, preds_dict):
        annos = []
        start = time.time()
        cls_preds_all = preds_dict["cls_preds"].squeeze(0)
        box_preds_all = preds_dict["box_preds"].squeeze(0)
        dir_preds_all = preds_dict["dir_preds"].squeeze(0)
        anchors_mask = example["anchors_mask"].squeeze(0)
        name_list, location_list, dimensions_list, rotation_y_list, score_list = [], [], [], [], []

        # box_preds_all = box_torch_ops.box_decode(box_preds_all, self.anchors)  ## 0.0025
        torch.cuda.synchronize()
        p1 = time.time()

        for cls, a_range in self.class_masks.items():
            a_mask = anchors_mask[a_range[0]: a_range[1]]
            box_preds = box_preds_all[a_range[0]: a_range[1]]
            cls_preds = cls_preds_all[a_range[0]: a_range[1]]
            dir_preds = dir_preds_all[a_range[0]: a_range[1]]
            anchors = self.anchors[a_range[0]: a_range[1]]

            ## Anchor mask
            # all = torch.cat([box_preds, cls_preds, dir_preds], dim=-1)[a_mask]
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
            anchors = anchors[a_mask]

            cls_scores = torch.sigmoid(cls_preds)
            top_scores = torch.max(cls_scores, dim=-1)[0]
            dir_labels = torch.max(dir_preds, dim=-1)[1]

            selected = None
            ## Score mask
            top_scores_keep = top_scores >= self._nms_score_threshold

            if top_scores_keep.any():
                top_scores = top_scores[top_scores_keep]
                box_preds = box_preds[top_scores_keep]
                dir_labels = dir_labels[top_scores_keep]
                anchors = anchors[top_scores_keep]

                if self._nms_pre_max_size is not None:
                    num_keeped_scores = top_scores.shape[0]
                    pre_max_size = min(num_keeped_scores, self._nms_pre_max_size)
                    top_scores, indices = torch.topk(top_scores, k=pre_max_size)

                    box_preds = box_preds[indices]
                    dir_labels = dir_labels[indices]
                    anchors = anchors[indices]

                    box_preds_np = box_preds.cpu().numpy()
                    dir_labels_np = dir_labels.cpu().numpy()
                    anchors_np = anchors.cpu().numpy()
                    top_scores_np = top_scores.cpu().numpy()

                    box_preds = box_torch_ops.box_decode(box_preds, anchors)
                    box_preds_np = box_np_ops.box_decode(box_preds_np, anchors_np)

                    # comparsion = np.fabs(box_preds.cpu().numpy() - box_preds_np)
                    # comparsion = comparsion.max()
                    # print('\nmax: ', comparsion)
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                boxes_for_nms_np = box_preds_np[:, [0, 1, 3, 4, 6]]

                torch.cuda.synchronize()
                p2 = time.time()  ### 0.00036

                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
                boxes_for_nms_test = boxes_for_nms.cpu().numpy()

                box_preds_corners_np = box_np_ops.center_to_corner_box2d(
                    boxes_for_nms_np[:, :2], boxes_for_nms_np[:, 2:4], boxes_for_nms_np[:, 4])
                boxes_for_nms_np = box_np_ops.corner_to_standup_nd_jit(box_preds_corners_np)
                '''
                comparsion = np.fabs(boxes_for_nms_test - boxes_for_nms_np)
                comparsion = comparsion.max()
                print('\nmax: ', comparsion)
                '''
                torch.cuda.synchronize()
                p3 = time.time()
                selected = nms2(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                )

                selected_np = nms(
                    boxes_for_nms_np,
                    top_scores_np,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                )
                # comparsion = selected == selected_np
                # print('\ncomparsion: ', comparsion.all())

            torch.cuda.synchronize()
            p4 = time.time()
            selected = torch.from_numpy(selected).long().to(self.device)
            if selected is not None:
                ## NMS mask
                box_preds = box_preds[selected]
                scores_preds = top_scores[selected]

                box_preds_np = box_preds_np[selected_np]
                scores_preds_np = top_scores_np[selected_np]

                # comparsion = np.fabs(box_preds.cpu().numpy() - box_preds_np)
                # comparsion = comparsion.max()
                # print('\nbox_preds max: ', comparsion)

                if self._use_direction_classifier:
                    dir_preds = dir_labels[selected].bool()
                    dir_mask = (box_preds[..., -1] > 0)
                    opp_labels = dir_mask ^ dir_preds

                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))

                    dir_preds_np = dir_labels_np[selected_np].astype(np.bool)
                    dir_mask_np = (box_preds_np[..., -1] > 0)
                    opp_labels_np = dir_mask_np ^ dir_preds_np
                    box_preds_np[..., -1] += np.where(opp_labels_np, np.pi, 0.0)

                scores_preds = scores_preds.detach().cpu().numpy()
                box_preds = box_preds.detach().cpu().numpy()

                # comparsion = np.fabs(box_preds - box_preds_np)
                # comparsion = comparsion.max()
                # print('\nbox_preds max: ', comparsion)

                limit_range = self.center_limit
                min_mask = np.any(box_preds_np[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds_np[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask

                ## Range mask
                box_preds = box_preds_np[range_mask]
                box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                scores_preds = scores_preds_np[range_mask]

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    name_list.append(np.full(dt_num, cls, dtype='<U10'))
                    location_list.append(box_preds[:, :3])
                    dimensions_list.append(box_preds[:, 3:6])
                    rotation_y_list.append(box_preds[:, 6])
                    score_list.append(scores_preds)

                '''
                limit_range = self.center_limit
                min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    name_list.append(np.full(dt_num, cls, dtype='<U10'))
                    location_list.append(box_preds[:, :3])
                    dimensions_list.append(box_preds[:, 3:6])
                    rotation_y_list.append(box_preds[:, 6])
                    score_list.append(scores_preds)
                '''

        anno = get_start_result_anno()
        if len(name_list) > 0:
            anno["name"] = np.concatenate(name_list)
            anno["location"] = np.concatenate(location_list)
            anno["dimensions"] = np.concatenate(dimensions_list)
            anno["rotation_y"] = np.concatenate(rotation_y_list)
            anno["score"] = np.concatenate(score_list)

        # anno["image_idx"] = np.array([img_id] * dt_num)
        annos.append(anno)
        torch.cuda.synchronize()
        p5 = time.time()
        '''
        print('p1', p1 - start)
        print('p2', p2 - p1)
        print('p3', p3 - p2)
        print('p4', p4 - p3)
        print('p5', p5 - p4)
        print('all', p5 - start)
        '''
        return annos

    def infer_v1(self, example, preds_dict):
        annos = []
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        batch_size = batch_box_preds.shape[0]

        batch_box_preds = batch_box_preds.view(batch_size, -1, self._box_code_size)
        batch_cls_preds = batch_cls_preds.view(batch_size, -1, self._num_class)

        batch_anchors = torch.stack([self.anchors] * batch_size)
        batch_box_preds = box_torch_ops.box_decode(batch_box_preds, batch_anchors)

        batch_anchors_mask = example["anchors_mask"]
        batch_dir_preds = preds_dict["dir_preds"]
        batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)

        for box_preds_all, cls_preds_all, dir_preds_all, a_mask_all in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_anchors_mask
        ):
            name_list, location_list, dimensions_list, rotation_y_list, score_list = [], [], [], [], []
            for cls, a_range in self.class_masks.items():
                a_mask = a_mask_all[a_range[0]: a_range[1]]
                box_preds = box_preds_all[a_range[0]: a_range[1]]
                cls_preds = cls_preds_all[a_range[0]: a_range[1]]
                dir_preds = dir_preds_all[a_range[0]: a_range[1]]

                ## Anchor mask
                box_preds = box_preds[a_mask]
                cls_preds = cls_preds[a_mask]
                dir_preds = dir_preds[a_mask]

                cls_scores = torch.sigmoid(cls_preds)
                top_scores = torch.max(cls_scores, dim=-1)[0]
                dir_labels = torch.max(dir_preds, dim=-1)[1]

                thresh = torch.tensor([self._nms_score_threshold], device=top_scores.device).type_as(top_scores)
                selected = None

                ## Score mask
                top_scores_keep = top_scores >= thresh
                if top_scores_keep.any():
                    top_scores = top_scores[top_scores_keep]
                    box_preds = box_preds[top_scores_keep]
                    # if self._use_direction_classifier:
                    dir_labels = dir_labels[top_scores_keep]

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

                if selected is not None:
                    ## NMS mask
                    box_preds = box_preds[selected]
                    scores_preds = top_scores[selected]
                    # name_inds = name_inds[selected]
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

                    limit_range = self.center_limit
                    min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                    max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                    range_mask = min_mask & max_mask

                    ## Range mask
                    box_preds = box_preds[range_mask]
                    box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                    scores_preds = scores_preds[range_mask]

                    dt_num = box_preds.shape[0]
                    if dt_num > 0:
                        name_list.append(np.full(dt_num, cls, dtype='<U10'))
                        location_list.append(box_preds[:, :3])
                        dimensions_list.append(box_preds[:, 3:6])
                        rotation_y_list.append(box_preds[:, 6])
                        score_list.append(scores_preds)

            anno = get_start_result_anno()
            if len(name_list) > 0:
                anno["name"] = np.concatenate(name_list)
                anno["location"] = np.concatenate(location_list)
                anno["dimensions"] = np.concatenate(dimensions_list)
                anno["rotation_y"] = np.concatenate(rotation_y_list)
                anno["score"] = np.concatenate(score_list)

            # anno["image_idx"] = np.array([img_id] * dt_num)
            annos.append(anno)
        return annos

    def infer_v2(self, example, preds_dict):
        annos = []
        start = time.time()
        cls_preds_all = preds_dict["cls_preds"].squeeze(0)
        box_preds_all = preds_dict["box_preds"].squeeze(0)
        dir_preds_all = preds_dict["dir_preds"].squeeze(0)
        anchors_mask = example["anchors_mask"].squeeze(0)
        name_list, location_list, dimensions_list, rotation_y_list, score_list = [], [], [], [], []

        torch.cuda.synchronize()
        p1 = time.time()

        for cls, a_range in self.class_masks.items():
            a_mask = anchors_mask[a_range[0]: a_range[1]]
            box_preds = box_preds_all[a_range[0]: a_range[1]]
            cls_preds = cls_preds_all[a_range[0]: a_range[1]]
            dir_preds = dir_preds_all[a_range[0]: a_range[1]]
            anchors = self.anchors[a_range[0]: a_range[1]]

            ## Anchor mask
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
            anchors = anchors[a_mask]

            cls_scores = torch.sigmoid(cls_preds)
            top_scores = torch.max(cls_scores, dim=-1)[0]
            dir_labels = torch.max(dir_preds, dim=-1)[1]

            selected = None
            ## Score mask
            top_scores_keep = top_scores >= self._nms_score_threshold

            if top_scores_keep.any():
                top_scores = top_scores[top_scores_keep]
                box_preds = box_preds[top_scores_keep]
                dir_labels = dir_labels[top_scores_keep]
                anchors = anchors[top_scores_keep]

                if self._nms_pre_max_size is not None:
                    num_keeped_scores = top_scores.shape[0]
                    pre_max_size = min(num_keeped_scores, self._nms_pre_max_size)
                    top_scores, indices = torch.topk(top_scores, k=pre_max_size)
                    box_preds = box_preds[indices]
                    dir_labels = dir_labels[indices]
                    anchors = anchors[indices]

                    box_preds_np = box_preds.cpu().numpy()
                    box_preds2 = box_np_ops.box_decode(box_preds_np, anchors.cpu().numpy())  ## 0.0025

                    box_preds = box_torch_ops.box_decode(box_preds, anchors)  ## 0.0025
                    box_preds1 = box_preds.cpu().numpy()

                    comparsion = np.fabs(box_preds1 - box_preds2)
                    comparsion = comparsion.max()
                    print('\nmax: ', comparsion)
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]

                torch.cuda.synchronize()
                p2 = time.time()  ### 0.00036

                box_preds_corners = box_torch_ops.center_to_corner_box2d(
                    boxes_for_nms[:, :2], boxes_for_nms[:, 2:4], boxes_for_nms[:, 4])
                boxes_for_nms = box_torch_ops.corner_to_standup_nd(box_preds_corners)
                torch.cuda.synchronize()
                p3 = time.time()
                selected = nms2(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                ).to(self.device)
            torch.cuda.synchronize()
            p4 = time.time()
            if selected is not None:
                ## NMS mask
                box_preds = box_preds[selected]
                scores_preds = top_scores[selected]
                # name_inds = name_inds[selected]
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

                limit_range = self.center_limit
                min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask

                ## Range mask
                box_preds = box_preds[range_mask]
                box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                scores_preds = scores_preds[range_mask]

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    name_list.append(np.full(dt_num, cls, dtype='<U10'))
                    location_list.append(box_preds[:, :3])
                    dimensions_list.append(box_preds[:, 3:6])
                    rotation_y_list.append(box_preds[:, 6])
                    score_list.append(scores_preds)

        anno = get_start_result_anno()
        if len(name_list) > 0:
            anno["name"] = np.concatenate(name_list)
            anno["location"] = np.concatenate(location_list)
            anno["dimensions"] = np.concatenate(dimensions_list)
            anno["rotation_y"] = np.concatenate(rotation_y_list)
            anno["score"] = np.concatenate(score_list)

        # anno["image_idx"] = np.array([img_id] * dt_num)
        annos.append(anno)
        torch.cuda.synchronize()
        p5 = time.time()
        '''
        print('p1', p1 - start)
        print('p2', p2 - p1)
        print('p3', p3 - p2)
        print('p4', p4 - p3)
        print('p5', p5 - p4)
        print('all', p5 - start)
        '''
        return annos


def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    dets_np = np.concatenate([bboxes, scores[:, np.newaxis]], axis=1)
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    else:
        return keep


def nms_torch(bboxes,
              scores,
              pre_max_size=None,
              post_max_size=None,
              iou_threshold=0.5):
    dets = torch.cat([bboxes, scores.unsqueeze(-1).float()], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(nms_gpu(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return None
    else:
        return torch.from_numpy(keep).long()


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


def nms_v1(bboxes,
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
        keep = torch.from_numpy(keep).long()
        return indices[keep]
    else:
        return torch.from_numpy(keep).long()


'''
class Inference:
    def __init__(self, config, anchor_assigner):
        self.anchors = torch.from_numpy(anchor_assigner.anchors).cuda()
        self.anchor_index = torch.from_numpy(np.arange(anchor_assigner.anchors.shape[0])).cuda()
        self.names = anchor_assigner.names
        self._nms_pre_max_size = 900
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
            ## Anchor mask
            box_preds = box_preds[a_mask]
            cls_preds = cls_preds[a_mask]
            dir_preds = dir_preds[a_mask]
            name_inds = self.anchor_index[a_mask]

            cls_scores = torch.sigmoid(cls_preds)
            top_scores = torch.max(cls_scores, dim=-1)[0]
            dir_labels = torch.max(dir_preds, dim=-1)[1]
            thresh = torch.tensor([self._nms_score_threshold], device=top_scores.device).type_as(top_scores)
            selected = None

            ## Score mask
            top_scores_keep = top_scores >= thresh
            if top_scores_keep.any():
                top_scores = top_scores[top_scores_keep]
                box_preds = box_preds[top_scores_keep]
                name_inds = name_inds[top_scores_keep]
                # if self._use_direction_classifier:
                dir_labels = dir_labels[top_scores_keep]

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
                ## NMS mask
                box_preds = box_preds[selected]
                scores_preds = top_scores[selected]
                name_inds = name_inds[selected]
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
                name_inds = name_inds.cpu().numpy()

                limit_range = self.center_limit
                min_mask = np.any(box_preds[:, :3] > limit_range[:3], axis=1)
                max_mask = np.any(box_preds[:, 3:6] < limit_range[3:], axis=1)
                range_mask = min_mask & max_mask
                ## Range mask
                box_preds = box_preds[range_mask]
                box_preds[..., -1] = box_np_ops.limit_period(box_preds[..., -1], period=2 * np.pi)
                scores_preds = scores_preds[range_mask]
                name_inds = name_inds[range_mask]

                dt_num = box_preds.shape[0]
                if dt_num > 0:
                    anno["name"] = self.names[name_inds]
                    anno["location"] = box_preds[:, :3]
                    anno["dimensions"] = box_preds[:, 3:6]
                    anno["rotation_y"] = box_preds[:, 6]
                    anno["score"] = scores_preds
                anno["image_idx"] = np.array([img_id] * dt_num)
            annos.append(anno)
        return annos
'''
