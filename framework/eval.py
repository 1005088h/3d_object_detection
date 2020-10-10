import numpy as np
from numba import cuda
import numba
import math
from framework.iou import rotate_iou_gpu_eval

def clean_data(gt_anno, dt_anno, current_class):
    ignored_gt, ignored_dt = [], []
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        if(gt_anno["name"][i].lower() == current_class):
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(-1)

    for i in range(num_dt):
        if(dt_anno["name"][i].lower() == current_class):
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)
    return num_valid_gt, ignored_gt, ignored_dt

def bev_box_overlap(boxes, qboxes, criterion=-1):
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou

def get_split_parts(num, num_part):
    same_part = num // num_part
    remain_num = num % num_part
    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]

def calculate_iou_partly(gt_annos, dt_annos, num_parts=50):
    num_examples = len(gt_annos)
    split_parts = get_split_parts(num_examples, num_parts)
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)
    parted_overlaps = []
    example_idx = 0
    for num_part in split_parts:

        gt_annos_part = gt_annos[example_idx:example_idx + num_part]
        dt_annos_part = dt_annos[example_idx:example_idx + num_part]

        loc = np.concatenate(
            [a["location"][:, :2] for a in gt_annos_part], 0)
        dims = np.concatenate(
            [a["dimensions"][:, :2] for a in gt_annos_part], 0)
        rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
        gt_boxes = np.concatenate(
            [loc, dims, -rots[..., np.newaxis]], axis=1)
        loc = np.concatenate(
            [a["location"][:, :2] for a in dt_annos_part], 0)
        dims = np.concatenate(
            [a["dimensions"][:, :2] for a in dt_annos_part], 0)
        rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
        dt_boxes = np.concatenate(
            [loc, dims, -rots[..., np.newaxis]], axis=1)
        overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(
            np.float64)

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

    return overlaps, parted_overlaps, total_gt_num, total_dt_num, split_parts


def _prepare_data(gt_annos, dt_annos, current_class):
    ignored_gts, ignored_dets, dt_score_list = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        num_valid_gt, ignored_gt, ignored_det = clean_data(gt_annos[i], dt_annos[i], current_class)
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        dt_score_list.append(dt_annos[i]["score"])
        total_num_valid_gt += num_valid_gt

    return ignored_gts, ignored_dets, dt_score_list, total_num_valid_gt

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
        # recall = l_recall
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    # print(len(thresholds), len(scores), num_gt)
    return thresholds

@numba.jit(nopython=True)
def compute_statistics_jit(overlaps, ignored_gt, ignored_det, dt_scores, min_overlap, thresh=0, compute_fp=False):

    det_size = ignored_det.size
    gt_size = ignored_gt.size
    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    thresh = thresh - 2.220446049250313e-16
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
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
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                det_idx = j
                valid_detection = dt_score

            elif (compute_fp and (overlap > min_overlap) and (overlap > max_overlap)):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1

        if (valid_detection == NO_DETECTION):
            fn += 1
        else:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not(assigned_detection[i] or ignored_det[i] == -1 or ignored_threshold[i])):
                fp += 1

    return tp, fp, fn, thresholds[:thresh_idx]

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

def eval_AP(gt_annos, dt_annos, current_classes, min_overlaps):
    
    assert len(gt_annos) == len(dt_annos)

    rets = calculate_iou_partly(dt_annos, gt_annos)# dt vs gt overlap
    overlaps, parted_overlaps, total_dt_num, total_gt_num, split_parts = rets
    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    precision = np.zeros([N_SAMPLE_PTS])
    recall = np.zeros([N_SAMPLE_PTS])

    for m, current_class in enumerate(current_classes):
        # ignored_gts, ignored_dets(maskes for current class) : 0 for current class, -1 for ignore class
        ignored_gts, ignored_dets, dt_score_list, total_num_valid_gt = _prepare_data(gt_annos, dt_annos, current_class)
        thresholdss = []
        for i in range(len(gt_annos)):
            # assign dt to gt based on highest score and min_overlap and return the scores list/thresholdss
            rets = compute_statistics_jit(
                overlaps[i],
                ignored_gts[i],
                ignored_dets[i],
                dt_score_list[i].astype('float32'),
                min_overlap=min_overlaps[0],
                thresh=0.0,
                compute_fp=False)
            tp, fp, fn, thresholds = rets
            thresholdss += thresholds.tolist()
        thresholdss = np.array(thresholdss)
        # get scores on 41 recall positions, every 1/40 of number of gt take one score
        thresholds = get_thresholds(thresholdss, total_num_valid_gt)
        thresholds = np.array(thresholds)
        pr = np.zeros([len(thresholds), 3])
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
                dt_scores.astype('float32'),
                min_overlap=min_overlaps[0],
                thresholds=thresholds)
            idx += num_part

        for i in range(len(thresholds)):
            recall[i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
            precision[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
        for i in range(len(thresholds)):
            precision[i] = np.max(precision[i:], axis=-1)
            recall[i] = np.max(recall[i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision
    }
    return ret_dict

def get_mAP_v2(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[i]
    return sums / 11 * 100
   
def get_eval_result(gt_annos, dt_annos, class_names=['vehicle']):
    ret = eval_AP(gt_annos, dt_annos, class_names, min_overlaps=[0.5])
    mAP_bev = get_mAP_v2(ret["precision"])
    return mAP_bev, ret["precision"]
