import onnx

import torch
import os
import time
import json
import copy
from framework.voxel_generator import VoxelGenerator
from framework.anchor_assigner import AnchorAssigner
from framework.loss_generator import LossGenerator
from framework.dataset import GenericDataset, InferData
from framework.metrics import Metric
from framework.inference import Inference
from framework.utils import merge_second_batch, worker_init_fn, example_convert_to_torch
# from networks.pointpillars8_mul import PointPillars
from networks.pointpillars5 import PointPillars
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from eval.eval import get_official_eval_result



def export():
    with open('configs/ntusl_20cm.json', 'r') as f:
        config = json.load(f)
    cuda_id = config['device']
    device = torch.device("cuda:" + cuda_id)
    config['device'] = device
    VoxelGenerator(config)
    net = PointPillars(config)

    model_path = Path(config['data_root']) / config['model_path'] / config['experiment']
    latest_model_path = os.path.join(model_path, 'latest.pth')

    checkpoint = torch.load(latest_model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('model loaded')
    print("num_trainable parameters:", len(list(net.parameters())))

    ONNX_FILE_PATH = 'pointpillars.onnx'
    torch.onnx.export(net, input, ONNX_FILE_PATH, input_names=['input'], output_names=['output'], export_params=True)
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)
