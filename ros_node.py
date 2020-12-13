#!/usr/bin/env python

import torch
import rospy
import numpy as np
import sys
import cv2
import os
import time
import json
from pathlib import Path
from framework.voxel_generator import VoxelGenerator
from framework.anchor_assigner import AnchorAssigner
from framework.dataset import InferData
from framework.inference import Inference
from networks.pointpillars import PointPillars
from framework.box_np_ops import points_in_rbbox
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import threading, queue

'''
from visualization_msgs.msg import Marker, MarkerArray
from kxr_msgs.msg import msg_roadobjects, msg_object3d
from ml_msgs.msg import ImagesArray
from geometry_msgs.msg import Point, Point32
from std_msgs.msg import MultiArrayDimension
'''

class PointPillarsNode(object):
    def __init__(self):
        print('initializing model...')
        # build model and preprocessor #
        with open('configs/ntusl_20cm.json', 'r') as f:
            config = json.load(f)

        device = torch.device("cuda:0")
        config['device'] = device
        self.voxel_generator = VoxelGenerator(config)
        self.anchor_assigner = AnchorAssigner(config)
        self.inference = Inference(config, self.anchor_assigner)
        self.infer_data = InferData(config, self.voxel_generator, self.anchor_assigner, torch.float32)
        self.net = PointPillars(config)
        self.net.cuda()
        model_path = Path(config['data_root']) / config['model_path'] / config['experiment']
        latest_model_path = model_path / 'latest.pth'
      
        checkpoint = torch.load(latest_model_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print('model loaded')
        self.net.eval()
        self.q_msg = queue.Queue(maxsize=2)

    
    def lidar_callback(self, msg):
        
        points = np.asarray( list( pc2.read_points(msg) ) )[:,:4].astype(np.float32)
        stamp = msg.header.stamp
        self.q_msg.put( (points, stamp) )
        
    def spin(self):
        time_elapse = 0.0
        len_infos = 0
        rospy.init_node("PointPillars", anonymous=False)
        rospy.Subscriber('/combined_lidar', PointCloud2, callback=self.lidar_callback, queue_size=1)

        print('spinning.')
        
        with torch.no_grad():
            while not rospy.is_shutdown():
                points, stamp = self.q_msg.get()
                start_time = time.time()
                example = self.infer_data.get(points)
                preds_dict = self.net(example)
                annos = self.inference.infer(example, preds_dict)
                dur = time.time() - start_time   
                time_elapse += dur
                len_infos += 1
                if len_infos >= 713:
                    break
                    
            print("infor len", len_infos)
            print("average time : %.5f" % (time_elapse / len_infos))

if __name__ == '__main__':
    PointPillarsNode().spin()

    

