import copy
import torch
import pickle
import os
import time
import json

from framework.voxel_generator import VoxelGenerator
from framework.anchor_assigner import AnchorAssigner
from framework.loss_generator import LossGenerator
from framework.dataset import GenericDataset
from framework.metrics import Metric
from framework.inference import Inference
from framework.utils import merge_second_batch, worker_init_fn, example_convert_to_torch
from networks.pointpillars import PointPillars
import numpy as np
import matplotlib.pyplot as plt
from eval.eval import get_official_eval_result


def infer():
    with open('configs/inhouse.json', 'r') as f:
        config = json.load(f)
    device = torch.device("cuda:0")
    config['device'] = device
    voxel_generator = VoxelGenerator(config)
    anchor_assigner = AnchorAssigner(config)
    inference = Inference(config, anchor_assigner)

    eval_dataset = GenericDataset(config, config['eval_info'], voxel_generator, anchor_assigner, training=False, augm=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True,
        collate_fn=merge_second_batch)

    net = PointPillars(config)
    net.cuda()

    model_path = config['model_path']
    experiment = config['experiment']
    model_path = os.path.join(model_path, experiment)
    latest_model_path = os.path.join(model_path, 'latest.pth')
    checkpoint = torch.load(latest_model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('model loaded')

    net.half()
    net.eval()
    dt_annos = []
    toGPU_t = 0.0
    network_t = 0.0
    post_t = 0.0
    data_iter = iter(eval_dataloader)
    for step in range(9999999):
        print('\rStep %d' % step, end='')
        try:
            example = next(data_iter)
            example = example_convert_to_torch(example, dtype=torch.float16)
            with torch.no_grad():
                preds_dict = net(example)
            dt_annos += inference.infer(example, preds_dict)
        except StopIteration:
            break

    gt_annos = [info["annos"] for info in eval_dataset.infos]
    eval_classes = ["vehicle", "pedestrian", "cyclist"]
    APs = get_official_eval_result(gt_annos, dt_annos, eval_classes)

if __name__ == "__main__":
    infer()

