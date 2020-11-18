from framework.voxel_generator import VoxelGenerator
from framework.anchor_assigner import AnchorAssigner
from framework.dataset import InferData
from framework.inference import Inference
from networks.pointpillars import PointPillars
import torch
import json
import numpy as np

CONFIG_PATH = '/home/xy/ST/object3d_det/configs/inhouse.json'
MODEL_PATH = '/home/xy/ST/dataset/inhouse/models/ntu_sl/120000.pth'

if __name__ == '__main__':

    with open(CONFIG_PATH,'r') as f:
        config = json.load(f)
    config['device'] = torch.device("cuda:0")
    voxel_generator = VoxelGenerator(config)
    anchor_assigner = AnchorAssigner(config)
    inference = Inference(config, anchor_assigner)
    infer_data = InferData(config, voxel_generator, anchor_assigner, torch.float16) # torch.float32, if torch.float16 not working
    model = PointPillars(config)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(torch.device("cuda:0")).half().eval()


    pts = np.load('/home/xy/ST/object3d_det/pts.npy')
    points = pts[:,:4]
    data = infer_data.get(points)
    with torch.no_grad():
        preds_dict = model(data)
    
    # print(preds_dict)
    dt_annos = inference.infer(data, preds_dict)
    print(dt_annos)
