import numba
import numpy as np
from numba import cuda


def nms_gpu(dets, nms_overlap_thresh, device_id=0):
    """nms in gpu. 
    
    Args:
        dets ([type]): [description]
        nms_overlap_thresh ([type]): [description]
        device_id ([type], optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """

    boxes_num = dets.shape[0]
    keep_out = np.zeros([boxes_num], dtype=np.int32)
    scores = dets[:, 4]
    order = scores.argsort()[::-1].astype(np.int32)
    boxes_host = dets[order, :]

    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    cuda.select_device(device_id)
    mask_host = np.zeros((boxes_num * col_blocks,), dtype=np.uint64)
    blockspergrid = (div_up(boxes_num, threadsPerBlock),
                     div_up(boxes_num, threadsPerBlock))
    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes_host.reshape([-1]), stream)
        mask_dev = cuda.to_device(mask_host, stream)
        nms_kernel[blockspergrid, threadsPerBlock, stream](
            boxes_num, nms_overlap_thresh, boxes_dev, mask_dev)
        mask_dev.copy_to_host(mask_host, stream=stream)
    # stream.synchronize()
    num_out = nms_postprocess(keep_out, mask_host, boxes_num)
    keep = keep_out[:num_out]
    return list(order[keep])


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba.jit(nopython=True)
def nms_postprocess(keep_out, mask_host, boxes_num):
    threadsPerBlock = 8 * 8
    col_blocks = div_up(boxes_num, threadsPerBlock)
    remv = np.zeros((col_blocks), dtype=np.uint64)
    num_to_keep = 0
    for i in range(boxes_num):
        nblock = i // threadsPerBlock
        inblock = i % threadsPerBlock
        mask = np.array(1 << inblock, dtype=np.uint64)
        if not (remv[nblock] & mask):
            keep_out[num_to_keep] = i
            num_to_keep += 1
            # unsigned long long *p = &mask_host[0] + i * col_blocks;
            for j in range(nblock, col_blocks):
                remv[j] |= mask_host[i * col_blocks + j]
                # remv[j] |= p[j];
    return num_to_keep


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def iou_device(a, b):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    width = max(right - left + 1, 0.)
    height = max(bottom - top + 1, 0.)
    interS = width * height
    Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return interS / (Sa + Sb - interS)


@cuda.jit('(int64, float32, float32[:], uint64[:])')
def nms_kernel(n_boxes, nms_overlap_thresh, dev_boxes, dev_mask):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.y
    col_start = cuda.blockIdx.x
    tx = cuda.threadIdx.x
    row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    dev_box_idx = threadsPerBlock * col_start + tx
    if tx < col_size:
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        cur_box_idx = threadsPerBlock * row_start + tx
        # cur_box = dev_boxes + cur_box_idx * 5;
        t = 0
        start = 0
        if row_start == col_start:
            start = tx + 1
        for i in range(start, col_size):
            iou = iou_device(dev_boxes[cur_box_idx * 5:cur_box_idx * 5 + 4],
                             block_boxes[i * 5:i * 5 + 4])
            if iou > nms_overlap_thresh:
                t |= 1 << i
        col_blocks = (n_boxes // threadsPerBlock + (
                n_boxes % threadsPerBlock > 0))
        dev_mask[cur_box_idx * col_blocks + col_start] = t



