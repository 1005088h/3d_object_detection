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
from framework.utils import merge_second_batch, worker_init_fn
from networks.pointpillars import PointPillars


def train(config_path=None):
    with open('configs/inhouse.json', 'r') as f:
        config = json.load(f)

    device = torch.device("cuda:0")
    config['device'] = device
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

    net = PointPillars(config)
    net.to(device)
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
        epoch = (step * config['batch_size']) // len(train_dataset) + 1
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
            print('epch %d, step %d' % (epoch, step))
            avg_loss = avg_loss / display_step
            avg_time = avg_time / display_step
            print('avg_loss', avg_loss)
            print('avg_time', avg_time)
            avg_loss = 0
            avg_time = 0
            print(metrics)
            metrics.clear()
        '''
        if step % eval_step == 0:
            net.eval()
            print("#################################")
            print("# EVAL")
            print("#################################")
            dt_annos = []

            for example in iter(eval_dataloader):
                preds_dict = net(example)
                dt_annos += inference.inter(example, preds_dict)
            
            gt_annos = [info["annos"] for info in eval_dataset._infos]

            result = get_official_eval_result(gt_annos, dt_annos, class_names, return_data=True)
            print(result)
            net.train()
            '''


def infer():
    with open('config.json', 'r') as f:
        config = json.load(f)
    device = torch.device("cuda:0")
    config['device'] = device
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
        collate_fn=merge_second_batch)

    net = PointPillars(config)
    net.cuda()
    model_path = config['model_path']
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.half()
    net.eval()
    dt_annos = []

    data_t = 0.0
    network_t = 0.0
    post_t = 0.0
    data_iter = iter(eval_dataloader)

    for step in range(9999999):
        print('step', step)
        try:
            t = time.time()
            example = next(data_iter)
            data_t += time.time() - t

            t = time.time()
            preds_dict = net(example)
            network_t += time.time() - t

            t = time.time()
            dt_annos += inference.infer(example, preds_dict)
            post_t += time.time() - t
        except StopIteration:
            break
    '''
    for step, example in enumerate(eval_dataloader):
        if step == 1:
            t = time.time()
        print("step", step)
        preds_dict = net(example)
        dt_annos += inference.infer(example, preds_dict)
    '''

    data_t = data_t / len(eval_dataset)
    network_t = network_t / len(eval_dataset)
    post_t = post_t / len(eval_dataset)

    print("avg_time : %.5f" % (data_t + network_t + post_t))
    print("data_t : %.5f" % data_t)
    print("network_t : %.5f" % network_t)
    print("post_t : %.5f" % post_t)

    with open(config['dt_info'], 'wb') as f:
        pickle.dump(dt_annos, f)
    '''   
    gt_annos = [
        info["annos"] for info in eval_dataset._infos
    ]
    '''


if __name__ == "__main__":
    train()
    #infer()
