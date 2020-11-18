import copy

import torch
import pickle
import os
import time
import json

from framework.voxel_generator import VoxelGenerator
from framework.anchor_assigner import AnchorAssigner
from framework.loss_generator import LossGenerator
from framework.dataset import GenericDataset, InferData
from framework.metrics import Metric
from framework.inference import Inference
from framework.utils import merge_second_batch, worker_init_fn, example_convert_to_torch
from networks.pointpillars import PointPillars
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from eval.eval import get_official_eval_result

def train(config_path=None):
    with open('configs/inhouse.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda:0")
    config['device'] = device
    voxel_generator = VoxelGenerator(config)
    anchor_assigner = AnchorAssigner(config)
    loss_generator = LossGenerator(config)
    metrics = Metric()
    inference = Inference(config, anchor_assigner)

    train_dataset = GenericDataset(config, config['train_info'], voxel_generator, anchor_assigner, training=True, augm=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True,
        collate_fn=merge_second_batch,
        worker_init_fn=worker_init_fn)

    eval_dataset = GenericDataset(config, config['eval_info'], voxel_generator, anchor_assigner, training=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True,
        collate_fn=merge_second_batch)
    eval_annos = [info["annos"] for info in eval_dataset.infos]

    net = PointPillars(config)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    step_num = 0

    model_path = Path(config['data_root']) / config['model_path'] / config['experiment']

    latest_model_path = os.path.join(model_path, 'latest.pth')
    log_file = os.path.join(model_path, 'log.txt')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    elif os.path.exists(latest_model_path):
        checkpoint = torch.load(latest_model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step_num = checkpoint['step']
        print('model loaded')

    print("num_trainable parameters:", len(list(net.parameters())))

    net.train()
    display_step = 50
    save_step = 5000
    eval_step = 5000
    avg_loss = 0

    data_iter = iter(train_dataloader)
    avg_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for step in range(step_num + 1, 10000000):
        epoch = (step * config['batch_size']) // len(train_dataset) + 1
        try:
            example = next(data_iter)
        except StopIteration:
            print("end epoch")
            data_iter = iter(train_dataloader)
            example = next(data_iter)

        optimizer.zero_grad()
        example = example_convert_to_torch(example)
        with torch.cuda.amp.autocast():
            preds_dict = net(example)
        loss_dict = loss_generator.generate(preds_dict, example)
        loss = loss_dict['loss']
        scaler.scale(loss).backward() #loss.backward()

        #torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        scaler.step(optimizer) #optimizer.step()
        scaler.update()
        labels = example['labels']
        cls_preds = preds_dict['cls_preds'].view(config['batch_size'], -1, 1)

        metrics.update(labels, cls_preds)
        avg_loss += loss.detach().item()
        if step % save_step == 0:
            torch.save({'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        latest_model_path)
            step_model_path = os.path.join(model_path, str(step) + '.pth')
            torch.save({'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        step_model_path)
            print("Model saved")

        if step % display_step == 0:
            avg_loss = avg_loss / display_step
            avg_time = (time.time() - avg_time) / display_step
            print('### Epoch %d, Step %d, Loss: %f, Time: %f' % (epoch, step, avg_loss, avg_time))
            print(metrics)
            metrics.clear()
            avg_loss = 0
            avg_time = time.time()

        if step % eval_step == 0:
            net.eval()
            print("#################################")
            print("# EVAL")
            print("#################################")
            dt_annos = []
            t = time.time()
            eval_total = len(eval_dataloader)
            for count, example in enumerate(eval_dataloader, start=1):
                print('\r%d / %d' % (count, eval_total), end='')
                example = example_convert_to_torch(example)
                preds_dict = net(example)
                dt_annos += inference.infer(example, preds_dict)
            t = (time.time() - t) / len(eval_dataloader)
            print('\nTime for each frame: %f\n' % t)

            gt_annos = copy.deepcopy(eval_annos)

            eval_classes = ["vehicle", "pedestrian", "cyclist"] #["vehicle", "pedestrian", "cyclist"]
            APs, eval_str = get_official_eval_result(gt_annos, dt_annos, eval_classes)
            log_str = 'Step: %d\n%s' % (step, eval_str)
            print(log_str)
            with open(log_file, 'a+') as f:
                f.write(log_str)
            net.train()


def evaluate():
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
        batch_size=None,
        shuffle=False,
        num_workers=3,
        pin_memory=False)


    print(len(eval_dataset))
    net = PointPillars(config)
    net.cuda()

    model_path = Path(config['data_root']) / config['model_path'] / config['experiment']
    latest_model_path = model_path / '120000.pth'
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
            example = merge_second_batch([example])
            t = time.time()
            example = example_convert_to_torch(example, dtype=torch.float16)
            torch.cuda.synchronize()
            d_t = time.time() - t
            toGPU_t += d_t
            # print(d_t)

            t = time.time()
            with torch.no_grad():
                preds_dict = net(example)
            torch.cuda.synchronize()
            d_t = time.time() - t
            network_t += d_t
            # print(d_t)

            t = time.time()
            dt_annos += inference.infer(example, preds_dict)
            torch.cuda.synchronize()
            d_t = time.time() - t
            post_t += d_t
            # print(d_t)
            # print(eval_dataset.load_t)
        except StopIteration:
            break

    load_t = eval_dataset.load_t / len(eval_dataset)
    voxelization_t = eval_dataset.voxelization_t / len(eval_dataset)
    toGPU_t = toGPU_t / len(eval_dataset)
    network_t = network_t / len(eval_dataset)
    post_t = post_t / len(eval_dataset)

    print("\navg_time : %.5f" % (voxelization_t + toGPU_t + network_t + post_t))
    print("load_t : %.5f" % load_t)
    print("voxelization_t : %.5f" % voxelization_t)
    print("toGPU_t : %.5f" % toGPU_t)
    print("network_t : %.5f" % network_t)
    print("post_t : %.5f" % post_t)

    dt_path = Path(config['data_root']) / config['experiment']
    if not os.path.exists(dt_path):
        os.makedirs(dt_path)

    with open(dt_path / config['dt_info'], 'wb') as f:
        pickle.dump(dt_annos, f)
    gt_annos = [info["annos"] for info in eval_dataset.infos]
    eval_classes = ["vehicle", "pedestrian", "cyclist"]  # ["vehicle", "pedestrian", "cyclist"]
    APs, eval_str = get_official_eval_result(gt_annos, dt_annos, eval_classes)
    print(eval_str)
    '''
    for i, AP in APs):
        precisions = ret["precision"]
        plt.axis([0, 1, 0, 1])
        recalls = np.linspace(0.0, 1.0, num=41)
        AP_str = "AP: %.4f " % AP
        print(AP_str, end=' ')
        plt.plot(recalls, precisions, label=AP_str)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('vehicle BEV AP@%.2f' % min_overlaps[i])
        plt.legend()
        plt.show()
    '''

def changeInfo(infos):
    for idx, info in enumerate(infos):
        if len(info['annos']['name']) > 0:
            difficulty_mask = info['annos']["num_points"] > 0
            for key in info['annos']:
                info['annos'][key] = info['annos'][key][difficulty_mask]
            car_mask = info['annos']['name'] == 'car'
            truck_mask = info['annos']['name'] == 'truck'
            bus_mask = info['annos']['name'] == 'bus'
            person_mask = info['annos']['name'] == 'person'
            bicycle_mask = info['annos']['name'] == 'bicycle'
            motorbike_mask = info['annos']['name'] == 'motorbike'
            vehicle_mask = car_mask | truck_mask | bus_mask
            info['annos']['name'][vehicle_mask] = "vehicle"
            pedestrian_mask = person_mask
            info['annos']['name'][pedestrian_mask] = "pedestrian"
            cyclist_mask = bicycle_mask | motorbike_mask
            info['annos']['name'][cyclist_mask] = "cyclist"

def infer():

    with open('configs/inhouse.json', 'r') as f:
        config = json.load(f)
    device = torch.device("cuda:0")
    config['device'] = device
    voxel_generator = VoxelGenerator(config)
    anchor_assigner = AnchorAssigner(config)
    inference = Inference(config, anchor_assigner)
    infer_data = InferData(config, voxel_generator, anchor_assigner, torch.float16)
    net = PointPillars(config)
    net.cuda()
    model_path = Path(config['data_root']) / config['model_path'] / config['experiment']
    latest_model_path = model_path / '120000.pth'
    checkpoint = torch.load(latest_model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('model loaded')
    net.half()
    net.eval()

    t = time.time()

    data_root = Path(config['data_root'])
    info_paths = config['eval_info']
    infos = []
    for info_path in info_paths:
        info_path = data_root / info_path
        with open(info_path, 'rb') as f:
            infos += pickle.load(f)
    changeInfo(infos)
    dt_annos = []
    for idx, info in enumerate(infos):

        # print('\ridx %d' % idx, end='')
        v_path = data_root / info['velodyne_path']
        points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 4])
        example = infer_data.get(points)
        with torch.no_grad():
            preds_dict = net(example)
        dt_annos += inference.infer(example, preds_dict)

    time_elapse = time.time() - t
    print("average time : %.5f" % (time_elapse / len(infos)))

    dt_path = Path(config['data_root']) / config['experiment']
    if not os.path.exists(dt_path):
        os.makedirs(dt_path)

    with open(dt_path / config['dt_info'], 'wb') as f:
        pickle.dump(dt_annos, f)
    gt_annos = [info["annos"] for info in infos]
    eval_classes = ["vehicle", "pedestrian", "cyclist"]  # ["vehicle", "pedestrian", "cyclist"]
    APs, eval_str = get_official_eval_result(gt_annos, dt_annos, eval_classes)
    print(eval_str)

if __name__ == "__main__":
    #train()
    #evaluate()
    infer()
