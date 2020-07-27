## Introduction
This repo is refactored from SECOND framework and make it more clean, readable and extendable. It decouple the loss and prediction from network, refactor the dataloader and remove unused features. Dataset bouding box coordinate system are converted to lidar coordinate system by info generator. The value order are: location: x, y, z. dimension: length, width and height. Rotaion in KITTI and nuScene are positive in clockwise refering to axis pointing to the left. infor generator convert the rotaion to anti-clockwise refering to lidar x axis. The rotaion operation in the framework are all positive in anti-clockwise ranging from -pi to pi.

## Environment
NOTE: Not all libs are necessary in the environment file
1. clone this repo
2. install the environment
```
conda env create -f environment.yml
```

## Dataset preparation
1. Convert or prepare dataset to kitti folder structure. nuScenes dataset has been converted and can be downloaded from the link below:
```
https://drive.google.com/drive/folders/1ei3YfRoSmS6koc1rdVrUAP_L6NY7wD-7?usp=sharing
```
2. Four folders can be found in the link: mini_train, mini_val, train and val. Down and put it under your favorite folder as  dataset_dir

3. Under info generator folder, generator data info from dataset_dir
```
python create_info.py create_info --data_path=dataset_dir
```
data_info.pkl will be generated under dataset_dir.

## Config
1. All configuration can be found in `config.json`. set the train_info and eval_info to the path of `data_info.pkl` created from datasets

2. Set the model_path to the model's path including the name where you want to save and load your model


## Training
```
python train.py
```

## Inference
uncomment the infer() and comment train() at the bottom of train.py
```
python train.py
```
`dt_info.pkl` will be created under the dt_info in `config.json`


## Network
Currently only contain pointpillars to verify the functionality of the dataloader framework.
Pointpillars can easily overfit mini_train datset. On the train dataset, after 10 hours training, it converge very low after loss reach 0.5 and it looks stuck there. The recall is around 0.5.

The output of the network is the feature map, hope it can be easier to insert other network.
PV-RCNN and SA-SSD can be added in to try.


## TODO
Add kitti evaluation. Difficulty depends on number of points. Other dataset doesn't have occluded and truncated info
Add multi-class support. Currenltly dataloader can load multi-class data but anchor generator and loss generator only support one class
Adapt to PV-RCNN and SA-SSD
Add and test data sampling
Add and test data data augmentation



