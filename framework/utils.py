import torch
from collections import defaultdict
import numpy as np
import time


def example_convert_to_torch(example, dtype=torch.float32):
    device = torch.device("cuda:0")
    example_torch = {}
    for k, v in example.items():
        if k in ["voxels"]:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "num_points_per_voxel"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.bool, device=device)
        else:
            example_torch[k] = v
    return example_torch

def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in ['voxels', 'num_points_per_voxel']:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'match_indices_num':
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            if len(batch_list) > 1:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(
                        coor, ((0, 0), (0, 1)),
                        mode='constant',
                        constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
            else:
                ret[key] = np.concatenate(elems, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])
