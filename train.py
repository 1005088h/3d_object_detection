import torch
import pickle
from voxel_generator import VoxelGenerator
from anchor_assigner import AnchorAssigner
from loss_generator import LossGenerator
import os
from dataset import GenericDataset
from networks.pointpillars import PointPillars
from metrics import Metric
from inference import Inference
from utils import _merge_second_batch, _worker_init_fn
import time
import json


def train(config_path=None):
    with open('config.json', 'r') as f:
        config = json.load(f)

    voxel_generator = VoxelGenerator(config)
    anchor_assigner = AnchorAssigner(config)
    loss_generator = LossGenerator(config)
    metrics = Metric()
    train_dataset = GenericDataset(config, config['train_info'], voxel_generator, anchor_assigner, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True,
        collate_fn=_merge_second_batch,
        worker_init_fn=_worker_init_fn)

    eval_dataset = GenericDataset(config, config['eval_info'], voxel_generator, anchor_assigner, training=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True,
        collate_fn=_merge_second_batch)

    net = PointPillars(config)
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    old_step = 1
    lowest_loss = 9999999

    model_path = config['model_path']
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        old_step = checkpoint['step'] + 1
        lowest_loss = checkpoint['loss']
        print('model loaded')

    # optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    print("num_trainable parameters:", len(list(net.parameters())))

    net.train()
    display_step = 50
    save_step = 500
    avg_loss = 0
    avg_time = 0

    data_iter = iter(train_dataloader)
    for step in range(old_step, 10000000):
        try:
            example = next(data_iter)
        except StopIteration:
            print("end epoch")
            data_iter = iter(train_dataloader)
            example = next(data_iter)

        optimizer.zero_grad()
        t = time.time()
        preds_dict = net(example)
        avg_time += time.time() - t
        loss_dict = loss_generator.generate(preds_dict, example)
        loss = loss_dict['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()

        labels = example['labels']
        cls_preds = preds_dict['cls_preds'].view(config['batch_size'], -1, len(config['detect_class']))
        metrics.update(labels, cls_preds)
        avg_loss += loss.detach().item()
        if step % save_step == 0 or avg_loss < lowest_loss:
            torch.save({'step': step,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': lowest_loss},
                       model_path)
            print("Model saved")
            if avg_loss < lowest_loss:
                lowest_loss = avg_loss

        if step % display_step == 0:
            print('step', step)
            avg_loss = avg_loss / display_step
            avg_time = avg_time / display_step
            print('avg_loss', avg_loss)
            print('avg_time', avg_time)
            avg_loss = 0
            avg_time = 0
            print(metrics)
            metrics.clear()


def infer():
    with open('config.json', 'r') as f:
        config = json.load(f)
    voxel_generator = VoxelGenerator(config)
    anchor_assigner = AnchorAssigner(config)
    inference = Inference(config, anchor_assigner)
    eval_dataset = GenericDataset(config, config['eval_info'], voxel_generator, anchor_assigner, training=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False,
        drop_last=True,
        collate_fn=_merge_second_batch)

    net = PointPillars(config)
    net.cuda()
    model_path = config['model_path']
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    dt_annos = []
    for step, example in enumerate(eval_dataloader):
        print("step", step)
        preds_dict = net(example)
        dt_annos += inference.infer(example, preds_dict)

    with open(config['dt_info'], 'wb') as f:
        pickle.dump(dt_annos, f)
    '''   
    gt_annos = [
        info["annos"] for info in eval_dataset._infos
    ]
    '''


if __name__ == "__main__":
    train()
    # infer()

