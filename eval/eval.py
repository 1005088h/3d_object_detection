import numpy as np
import numba
from .iou import rotate_iou_gpu_eval

def clean_data(gt_anno, dt_anno, current_class, num_points_thresh):
    ignored_gt, ignored_dt = [], []
    current_cls_name = current_class.lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()
        if gt_name == current_cls_name:
            if gt_anno["num_points"][i] == 0:
                ignored_gt.append(-1)
            elif gt_anno["num_points"][i] > num_points_thresh:
                ignored_gt.append(0)
                num_valid_gt += 1
            else:
                ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    return num_valid_gt, ignored_gt, ignored_dt


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall))
                and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds

@numba.jit(nopython=True)
def compute_statistics_jit(overlaps, ignored_gt, ignored_det, dt_scores, min_overlap, thresh=0, compute_fp=False):

    det_size = ignored_det.size
    gt_size = ignored_gt.size
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size

    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True

    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size,))
    thresh_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0

        for j in range(det_size):
            if ignored_det[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if not compute_fp and (overlap > min_overlap) and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score

            elif compute_fp and (overlap > min_overlap) and (overlap > max_overlap):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1

        if valid_detection == NO_DETECTION and ignored_gt[i] == 0:
            fn += 1
        elif valid_detection != NO_DETECTION and ignored_gt[i] == 1:
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if not(assigned_detection[i] or ignored_det[i] == -1 or ignored_threshold[i]):
                fp += 1

    return tp, fp, fn, thresholds[:thresh_idx]


#@numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True)
def d3_box_overlap_kernel_camera(boxes, qboxes, rinc, criterion=-1):
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


#@numba.jit(nopython=True, parallel=True)
@numba.jit(nopython=True)
def d3_box_overlap_kernel_lidar(boxes, qboxes, rinc, criterion=-1):
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 2] + boxes[i, 5] / 2, qboxes[j, 2] + qboxes[j, 5] / 2) - max(
                    boxes[i, 2] - boxes[i, 5] / 2, qboxes[j, 2] - qboxes[j, 5] / 2))
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

@numba.jit(nopython=True)
def fused_compute_statistics(overlaps,
                             pr,
                             gt_nums,
                             dt_nums,
                             ignored_gts,
                             ignored_dets,
                             dt_scores,
                             min_overlap,
                             thresholds):
    gt_num = 0
    dt_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:
                               gt_num + gt_nums[i]]

            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dt_score = dt_scores[dt_num:dt_num + dt_nums[i]]
            tp, fp, fn, _ = compute_statistics_jit(
                overlap,
                ignored_gt,
                ignored_det,
                dt_score,
                min_overlap=min_overlap,
                thresh=thresh,
                compute_fp=True)

            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn

        gt_num += gt_nums[i]
        dt_num += dt_nums[i]

def d3_box_overlap_camera(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel_camera(boxes, qboxes, rinc, criterion)
    return rinc

def d3_box_overlap_lidar(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 1, 3, 4, 6]],
                               qboxes[:, [0, 1, 3, 4, 6]], 2)
    d3_box_overlap_kernel_lidar(boxes, qboxes, rinc, criterion)
    return rinc

def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

def calculate_iou_partly_lidar(gt_annos, dt_annos, metric='bev', num_parts=50):
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    parted_overlaps = []
    example_idx = 0
    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 'bev':
            loc = np.concatenate([a["location"][:, :2] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, :2] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, -rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, :2] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, :2] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, -rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)
        elif metric == '3d':
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, -rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, -rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap_lidar(gt_boxes, dt_boxes).astype(np.float64)
        else:
            raise ValueError("unknown metric")

        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num#, split_parts

def calculate_iou_partly_camera(gt_annos, dt_annos, metric='bev', num_parts=50):

    assert len(gt_annos) == len(dt_annos)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    parted_overlaps = []
    example_idx = 0

    for num_part in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        if metric == 'bev':
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
                np.float64)
        elif metric == '3d':
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap_camera(gt_boxes, dt_boxes).astype(
                np.float64)
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += num_part
    overlaps = []
    example_idx = 0
    for j, num_part in enumerate(split_parts):
        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(num_part):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num,
                                   dt_num_idx:dt_num_idx + dt_box_num])
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += num_part

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def _prepare_data(gt_annos, dt_annos, current_class, num_points_thresh):
    ignored_gts, ignored_dets, dt_score_list = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_det = clean_data(gt_annos[i], dt_annos[i], current_class, num_points_thresh)
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        dt_score_list.append(dt_annos[i]["score"].astype('float32'))
        total_num_valid_gt += num_valid_gt

    return ignored_gts, ignored_dets, dt_score_list, total_num_valid_gt


def eval_class_AP(gt_annos,
                  dt_annos,
                  class_names,
                  metric,
                  min_overlaps,
                  frame,
                  num_points_thresh,
                  num_parts=50,
                  ):

    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)

    if frame == 'lidar':
        rets = calculate_iou_partly_lidar(dt_annos, gt_annos, metric, num_parts)
    else:
        rets = calculate_iou_partly_camera(dt_annos, gt_annos, metric, num_parts)

    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets
    N_SAMPLE_PTS = 41
    min_overlaps_t = list(min_overlaps.values())
    num_minoverlap = len(min_overlaps_t[0])
    num_class = len(class_names)
    precision = np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_minoverlap, N_SAMPLE_PTS])
    for m, current_class in enumerate(class_names):
        ignored_gts, ignored_dets, dt_score_list, total_num_valid_gt = _prepare_data(gt_annos, dt_annos, current_class, num_points_thresh)
        for k, min_overlap in enumerate(min_overlaps[current_class]):
            thresholdss = []
            for i in range(len(gt_annos)):
                rets = compute_statistics_jit(
                    overlaps[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    dt_score_list[i],
                    min_overlap=min_overlap,
                    thresh=0.0,
                    compute_fp=False)
                tp, fp, fn, thresholds = rets
                thresholdss += thresholds.tolist()
            thresholdss = np.array(thresholdss)
            thresholds = get_thresholds(thresholdss, total_num_valid_gt)
            thresholds = np.array(thresholds)
            pr = np.zeros([len(thresholds), 4])
            idx = 0
            for j, num_part in enumerate(split_parts):
                ignored_dets_part = np.concatenate(
                    ignored_dets[idx:idx + num_part], 0)
                ignored_gts_part = np.concatenate(
                    ignored_gts[idx:idx + num_part], 0)
                dt_scores = np.concatenate(
                    dt_score_list[idx:idx + num_part], 0)
                fused_compute_statistics(
                    parted_overlaps[j],
                    pr,
                    total_gt_num[idx:idx + num_part],
                    total_dt_num[idx:idx + num_part],
                    ignored_gts_part,
                    ignored_dets_part,
                    dt_scores,
                    min_overlap=min_overlap,
                    thresholds=thresholds)
                idx += num_part
            for i in range(len(thresholds)):
                recall[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                precision[m, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
            for i in range(len(thresholds)):
                precision[m, k, i] = np.max(precision[m, k, i:], axis=-1)
                # recall[m, k, i] = np.max(recall[m, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


'''
# Convert Lidar Frame to Camera frame to test in camera frame
    for anno in gt_annos + dt_annos:
        anno['location'][:, 2] -= anno['dimensions'][:, 2] / 2
        anno['location'] = anno['location'][:, (1, 2, 0)]
        anno['location'][:, 0:2] *= -1
        anno['dimensions'] = anno['dimensions'][:, (0, 2, 1)]
        anno['rotation_y'] = - anno['rotation_y'] - np.pi / 2
'''

def get_official_eval_result(gt_annos, dt_annos, class_names):
    min_overlaps = {'vehicle':    [0.7, 0.5],
                    'pedestrian': [0.5, 0.25],
                    'cyclist':    [0.5, 0.25]}

    metrics = ['bev', '3d']     # bev, 3d
    frame = 'lidar'             # lidar, camera
    num_point_threshold = 5     # 0, 5, 10
    results = []
    eval_str = ''
    for metric in metrics:
        eval_str += '\n#### Metric: %s, num_points > %d\n' % (metric, num_point_threshold)
        ret = eval_class_AP(gt_annos, dt_annos, class_names, metric, min_overlaps, frame, num_point_threshold)
        mAP = get_mAP(ret['precision'])
        results.append(mAP)
        for i, cls in enumerate(class_names):
            eval_str += cls + ':\t'
            ious = min_overlaps[cls]
            for j, iou in enumerate(ious):
                eval_str += '@%.2f %.4f\t' % (iou, mAP[i][j])
            eval_str += '\n'
    return results, eval_str
