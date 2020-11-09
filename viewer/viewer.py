import io as sysio
from pathlib import Path
import os
import pickle
import sys
import time
from functools import partial
import datetime
import numpy as np
import copy
from matplotlib.backends.backend_qt5agg import \
    NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QTextEdit,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget, QProgressBar)

from skimage import io

from bbox_plot import GLColor
import json

from framework import box_np_ops
from eval.eval import bev_box_overlap
from utils import remove_low_score
from views import MatPlotLibView, KittiDrawControl, KittiPointCloudView
from utils import Settings, riou3d_shapely


from framework.anchor_assigner import AnchorAssigner
from framework.voxel_generator import VoxelGenerator
from framework.dataset import GenericDataset

class PCViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_root = None
        self.title = 'PointCloudViewer'
        self.bbox_window = [10, 10, 1600, 900]
        self.sstream = sysio.StringIO()
        self.json_setting = Settings(str(Path.home() / ".kittiviewerrc"))
        self.init_ui()

        self.infos = None
        self.detection_annos = None
        self.current_idx = 0
        self.current_image = None
        self.points = None
        self.gt_boxes = None
        self.gt_names = None
        self.gt_bbox = None
        self.classes = ["vehicle", "pedestrian", "cyclist"]

        self.config_path = '../configs/inhouse.json'
        self.dataset = None
        self.anchors = None
        self.augm = False
        self.plot_anchors = False
        self.plot_voxel = False


    def build_dataset(self, info_path=None):

        with open(self.config_path, 'r') as f:
            config = json.load(f)
        voxel_generator = VoxelGenerator(config)
        anchor_assigner = AnchorAssigner(config)
        dataset = GenericDataset(config, info_path, voxel_generator, anchor_assigner, training=True, augm=self.augm)
        return dataset

    def init_ui(self):

        self.setWindowTitle(self.title)
        self.setGeometry(*self.bbox_window)

        control_panel_layout = QVBoxLayout()

        info_path = self.json_setting.get("latest_info_path", "")
        self.w_info_path = QLineEdit(info_path)
        idx = self.json_setting.get("idx", "0")
        self.w_idx = QLineEdit(idx)
        det_path = self.json_setting.get("latest_det_path", "")
        self.w_det_path = QLineEdit(det_path)
        layout = QFormLayout()
        layout.addRow(QLabel("info path:"), self.w_info_path)
        layout.addRow(QLabel("idx:"), self.w_idx)
        layout.addRow(QLabel("det path:"), self.w_det_path)
        self.w_config_gbox = QGroupBox("Read Config")
        self.w_config_gbox.setLayout(layout)
        control_panel_layout.addWidget(self.w_config_gbox)

        self.w_load = QPushButton('load info')
        self.w_load.clicked.connect(self.on_loadButtonPressed)
        self.w_load_det = QPushButton('load detection')
        self.w_load_det.clicked.connect(self.on_loadDetPressed)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.w_load)
        h_layout.addWidget(self.w_load_det)
        control_panel_layout.addLayout(h_layout)

        self.w_config = KittiDrawControl('ctrl')
        config = self.json_setting.get("config", "")
        if config != "":
            self.w_config.loads(config)
        self.w_config.configChanged.connect(self.on_configchanged)
        self.w_plot = QPushButton('plot')
        self.w_plot.clicked.connect(self.on_plotButtonPressed)
        self.w_show_panel = QPushButton('control panel')
        self.w_show_panel.clicked.connect(self.on_panel_clicked)
        control_panel_layout.addWidget(self.w_plot)
        control_panel_layout.addWidget(self.w_show_panel)

        self.w_next = QPushButton('next')
        self.w_next.clicked.connect(partial(self.on_nextOrPrevPressed, prev=False))
        self.w_prev = QPushButton('prev')
        self.w_prev.clicked.connect(partial(self.on_nextOrPrevPressed, prev=True))
        layout = QHBoxLayout()
        layout.addWidget(self.w_prev)
        layout.addWidget(self.w_next)
        control_panel_layout.addLayout(layout)

        simg_path = self.json_setting.get("save_image_path", "")
        self.w_image_save_path = QLineEdit(simg_path)
        svid_path = self.json_setting.get("save_video_path", "")
        self.w_video_save_path = QLineEdit(svid_path)
        layout = QFormLayout()
        layout.addRow(QLabel("image save path:"), self.w_image_save_path)
        layout.addRow(QLabel("video save path:"), self.w_video_save_path)
        self.w_save_gbox = QGroupBox("Save Config")
        self.w_save_gbox.setLayout(layout)
        control_panel_layout.addWidget(self.w_save_gbox)


        self.w_save_image = QPushButton('save image')
        self.w_save_image.clicked.connect(self.on_saveImagePressed)
        self.w_save_video = QPushButton('save video')
        self.w_save_video.clicked.connect(self.on_saveVideoPressed)
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.w_save_image)
        h_layout.addWidget(self.w_save_video)
        control_panel_layout.addLayout(h_layout)

        self.w_plt = MatPlotLibView()
        center_widget = QWidget(self)
        self.w_plt_toolbar = NavigationToolbar(self.w_plt, center_widget)
        plt_layout = QVBoxLayout()
        plt_layout.addWidget(self.w_plt)
        plt_layout.addWidget(self.w_plt_toolbar)
        control_panel_layout.addLayout(plt_layout)

        self.w_output = QTextEdit()
        control_panel_layout.addWidget(self.w_output)

        self.center_layout = QHBoxLayout()
        self.w_pc_viewer = KittiPointCloudView(self.w_config, coors_range=self.w_config.get("CoorsRange"))
        self.center_layout.addWidget(self.w_pc_viewer)
        self.center_layout.addLayout(control_panel_layout)
        self.center_layout.setStretch(0, 7)
        self.center_layout.setStretch(1, 3)
        center_widget.setLayout(self.center_layout)
        self.setCentralWidget(center_widget)
        self.show()

    def on_loadButtonPressed(self):
        info_path = [self.w_info_path.text()]
        self.dataset = self.build_dataset(info_path)
        self.log("load", len(self.dataset), "infos.")
        self.json_setting.set("latest_info_path", str(info_path))

    def on_loadDetPressed(self):
        det_path = self.w_det_path.text()
        if os.path.exists(det_path):
            with open(det_path, "rb") as f:
                dt_annos = pickle.load(f)
        else:
            dt_annos = []
        self.detection_annos = dt_annos
        self.log(f"load {len(dt_annos)} detections.")
        self.json_setting.set("latest_det_path", det_path)

    def on_plotButtonPressed(self):
        if self.dataset is None:
            self.error("you must load data Infos first.")
            return
        self.current_idx = int(self.w_idx.text())
        print(self.current_idx)
        self.plot_all(self.current_idx)


    def plot_all(self, idx):
        self.w_plt.reset_plot()
        self.load_info(idx)
        self.plot_image()
        self.plot_pointcloud()
        return True

    def load_info(self, idx):
        self.info = self.dataset.infos[idx]
        self.example = self.dataset[idx]
        if 'img_path' in self.info:
            img_path = self.dataset.root_dir / self.info['img_path']
            if img_path != "":
                self.current_image = io.imread(img_path)
            else:
                self.current_image = None
        else:
            self.current_image = None

        self.points = self.example['points']
        self.gt_names = self.example['annos']['gt_names']
        self.gt_boxes = self.example['annos']['gt_boxes']
        if self.plot_anchors:
            labels = self.example['labels']
            assigned_mask = (labels > 0)
            self.anchors = self.dataset.anchor_assigner.anchors[assigned_mask]
            '''
            anchor_mask = self.example['anchors_mask']
            self.anchors = self.dataset.anchor_assigner.anchors[anchor_mask]
            np.random.shuffle(self.anchors)
            self.anchors = self.anchors[:100]
            '''


    def plot_image(self):
        if self.current_image is not None:
            self.w_plt.ax.imshow(self.current_image)
            self.w_plt.draw()


    def plot_pointcloud(self):
        point_color = self.w_config.get("PointColor")[:3]
        point_color = (*point_color, self.w_config.get("PointAlpha"))
        point_color = np.tile(np.array(point_color), [self.points.shape[0], 1])

        point_size = np.full(
            [self.points.shape[0]],
            self.w_config.get("PointSize"),
            dtype=np.float32)

        self.w_pc_viewer.draw_bounding_box(self.w_config.get("CoorsRange"))
        if self.gt_boxes is not None and len(self.gt_boxes) > 0:
            gt_boxes = copy.deepcopy(self.gt_boxes)
            gt_boxes[:, 3:6] = gt_boxes[:, 3:6] + np.array([1.2, 0.8, 8])
            #gt_boxes[:, 2] = gt_boxes[:, 2] + np.array([1])
            gt_point_table = box_np_ops.points_in_rbbox(self.points, gt_boxes)
            self.gt_filled_mask = gt_point_table.sum(axis=0)
            gt_point_mask = gt_point_table.any(1)
            point_size[gt_point_mask] = self.w_config.get("GTPointSize")
            gt_point_color = self.w_config.get("GTPointColor")
            gt_point_color = (*gt_point_color[:3], self.w_config.get("GTPointAlpha"))
            point_color[gt_point_mask] = gt_point_color

        self.w_pc_viewer.remove("gt_boxes/labels")
        self.w_pc_viewer.remove("gt_boxes")
        if self.gt_names is not None and len(self.gt_names) > 0 and self.w_config.get("DrawGTBoxes"):
            self.plot_gt_boxes_in_pointcloud()

        self.w_pc_viewer.remove("dt_boxes/labels")
        self.w_pc_viewer.remove("dt_boxes")
        if self.detection_annos is not None and len(self.detection_annos) > 0 and self.w_config.get("DrawDTBoxes"):
            detection_anno = self.detection_annos[self.current_idx]
            self.draw_detection(detection_anno)

        self.w_pc_viewer.remove("anchors")
        if self.plot_anchors:
            self.plot_anchors_in_pointcloud()


        self.w_pc_viewer.scatter(
            "pointcloud", self.points[:, :3], point_color, size=point_size)

    def draw_detection(self, detection_anno, label_color=GLColor.Cyan):
        dt_box_color = self.w_config.get("DTBoxColor")[:3]
        dt_box_color = (*dt_box_color, self.w_config.get("DTBoxAlpha"))
        detection_anno = remove_low_score(detection_anno, self.w_config.get("DTScoreThreshold"))
        if detection_anno is not None and detection_anno['score'].shape[0] > 0:
            dims = detection_anno['dimensions']
            loc = detection_anno['location']
            rots = detection_anno['rotation_y']
            scores = detection_anno['score']
            label = detection_anno['name']

            #num_points = detection_anno['num_points']

            dt_box_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            dt_boxes_corners = box_np_ops.center_to_corner_box3d(
                dt_box_lidar[:, :3],
                dt_box_lidar[:, 3:6],
                dt_box_lidar[:, 6],
                origin=[0.5, 0.5, 0.5],
                axis=2)

            if self.gt_boxes is not None and self.gt_boxes.shape[0] > 0:
                '''
                iou_3d = riou3d_shapely(self.gt_boxes, dt_box_lidar)
                if iou_3d.shape[0] != 0:
                    dt_to_gt_box_3diou = iou_3d.max(0)
                else:
                    dt_to_gt_box_3diou = np.zeros([0, 0])
                '''
                iou_2d = bev_box_overlap(dt_box_lidar[:, [0, 1, 3, 4, 6]], self.gt_boxes[:, [0, 1, 3, 4, 6]])
                if iou_2d.shape[0] != 0:
                    dt_to_gt_box_iou = iou_2d.max(1)
                else:
                    dt_to_gt_box_iou = np.zeros([0, 0])
            num_dt = dt_box_lidar.shape[0]
            '''
            dt_boxes_corners_cam = box_np_ops.lidar_to_camera(dt_boxes_corners, rect, Trv2c)
            dt_boxes_corners_cam = dt_boxes_corners_cam.reshape((-1, 3))
            dt_boxes_corners_cam_p2 = box_np_ops.project_to_image(dt_boxes_corners_cam, P2)
            dt_boxes_corners_cam_p2 = dt_boxes_corners_cam_p2.reshape([-1, 8, 2])
            '''
            #print(len(label), len(scores),len(dt_box_lidar),len(dt_to_gt_box_iou))
            if self.gt_boxes is not None and self.gt_boxes.shape[0] > 0:
                dt_scores_text = [
                    # f'score={s:.2f}, iou={i:.2f}'
                    # for s, i in zip(label, dt_to_gt_box_iou)
                    f'label={l}, score={s:.2f}, x={x:.2f}, y={y:.2f}, iou_2d={iou:.2f}'
                    for i, (l, s, x, y, iou) in enumerate(zip(label, scores, dt_box_lidar[:, 0], dt_box_lidar[:, 1], dt_to_gt_box_iou))
                ]
            else:
                dt_scores_text = [
                    f'score={s:.2f}, x={x:.2f}, y={y:.2f}, r={r:.2f}'
                    for s, x, y, r in zip(scores, dt_box_lidar[:, 0], dt_box_lidar[:, 1], dt_box_lidar[:, 3])
                ]
            if self.w_config.get("DrawDTLabels"):
                self.w_pc_viewer.labels("dt_boxes/labels",
                                        dt_boxes_corners[:, 1, :], dt_scores_text,
                                        label_color, 15)

            dt_box_color = np.tile(np.array(dt_box_color)[np.newaxis, ...], [num_dt, 1])
            if self.w_config.get("DTScoreAsAlpha") and scores is not None:
                dt_box_color = np.concatenate([dt_box_color[:, :3], scores[..., np.newaxis]], axis=1)
            self.w_pc_viewer.boxes3d("dt_boxes", dt_boxes_corners, dt_box_color,
                                     self.w_config.get("DTBoxLineWidth"), 1.0)

    def plot_gt_boxes_in_pointcloud(self):
        if self.gt_names is not None and len(self.gt_names) > 0:
            gt_box_color = self.w_config.get("GTBoxColor")[:3]
            gt_box_color = (*gt_box_color, self.w_config.get("GTBoxAlpha"))
            #diff = self.difficulty.tolist()

            labels_ = [
                "%s, %.3f" % (i, bx[6]) for i, bx in zip(self.gt_names, self.gt_boxes)
            ]
            '''
            labels_ = [
                "%s:X%.2f Y%.2f Z%.2f L%.2f W%.2f H%.2f R%3f" % (i, bx[0], bx[1], bx[2], bx[3], bx[4], bx[5], bx[6]) for i, bx in zip(self.gt_names, self.gt_boxes)
            ]
            '''
            boxes_corners = box_np_ops.center_to_corner_box3d(
                self.gt_boxes[:, :3],
                self.gt_boxes[:, 3:6],
                self.gt_boxes[:, 6],
                origin=[0.5, 0.5, 0.5],
                axis=2)

            self.w_pc_viewer.boxes3d("gt_boxes", boxes_corners, gt_box_color,
                                     3.0, 1.0)

            if self.w_config.get("DrawGTLabels"):
                self.w_pc_viewer.labels("gt_boxes/labels", boxes_corners[:, 0, :],
                                        labels_, GLColor.Green, 15)

    def plot_anchors_in_pointcloud(self):
        if self.anchors is not None and len(self.anchors) > 0:
            anchors_color = self.w_config.get("GTBoxColor")[:3]
            anchors_color = (*anchors_color, self.w_config.get("GTBoxAlpha"))
            boxes_corners = box_np_ops.center_to_corner_box3d(
                self.anchors[:, :3],
                self.anchors[:, 3:6],
                self.anchors[:, 6],
                origin=[0.5, 0.5, 0.5],
                axis=2)
            self.w_pc_viewer.boxes3d("anchors", boxes_corners, anchors_color, 3.0, 1.0)



    def on_panel_clicked(self):
        if self.w_config.isHidden():
            self.w_config.show()
        else:
            self.w_config.hide()

    def message(self, value, *arg, color="Black"):
        colorHtml = f"<font color=\"{color}\">"
        endHtml = "</font><br>"
        msg = self.print_str(value, *arg)
        self.w_output.insertHtml(colorHtml + msg + endHtml)
        self.w_output.verticalScrollBar().setValue(
            self.w_output.verticalScrollBar().maximum())

    def error(self, value, *arg):
        time_str = datetime.datetime.now().strftime("[%H:%M:%S]")
        return self.message(time_str, value, *arg, color="Red")

    def log(self, value, *arg):
        time_str = datetime.datetime.now().strftime("[%H:%M:%S]")
        return self.message(time_str, value, *arg, color="Black")

    def warning(self, value, *arg):
        time_str = datetime.datetime.now().strftime("[%H:%M:%S]")
        return self.message(time_str, value, *arg, color="Yellow")

    def on_saveImagePressed(self):
        idx = int(self.w_idx.text())
        image_save_path = self.w_image_save_path.text()

        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        lidar_path = os.path.join(image_save_path, 'lidars')
        image_path = os.path.join(image_save_path, 'images')
        if not os.path.exists(lidar_path):
            os.makedirs(lidar_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)
        file_name = str(idx)
        lidar_path = os.path.join(lidar_path, file_name) + '.jpg'
        image_path = os.path.join(image_path, file_name) + '.jpg'

        self.json_setting.set("save_image_path", image_save_path)
        if self.current_image is not None:
            io.imsave(image_path, self.current_image)
        # p = self.w_pc_viewer.grab()
        self.w_pc_viewer.paintGL()
        p = self.w_pc_viewer.grabFrameBuffer()
        p.save(lidar_path, 'jpg')
        self.log("image saved to", image_path)
        self.log("image saved to", lidar_path)

    def on_nextOrPrevPressed(self, prev):
        info_len = len(self.dataset)
        if prev is True:
            self.current_idx = max(self.current_idx - 1, 0)
        else:
            self.current_idx = min(self.current_idx + 1, info_len - 1)
        self.w_idx.setText(str(self.current_idx))
        self.plot_all(self.current_idx)

    def on_saveVideoPressed(self):
        end_idx = self.current_idx + 189
        while self.current_idx < end_idx:
            print(self.current_idx)
            self.on_saveImagePressed()
            self.on_nextOrPrevPressed(False)

    def print_str(self, value, *arg):
        #self.strprint.flush()
        self.sstream.truncate(0)
        self.sstream.seek(0)
        print(value, *arg, file=self.sstream)
        return self.sstream.getvalue()


    def draw_gt_in_image(self):
        if self.info is None:
            self.error("you must load infos and choose a existing image idx first.")
            return
        if self.gt_boxes is None:
            return
        rect = self.info['calib/R0_rect']
        P2 = self.info['calib/P2']
        Trv2c = self.info['calib/Tr_velo_to_cam']
        imsize = self.info['img_shape']

        gt_boxes_camera = box_np_ops.box_lidar_to_camera(self.gt_boxes, rect, Trv2c)

        front_mask = gt_boxes_camera[:, 2] > 0
        gt_boxes_camera = gt_boxes_camera[front_mask]
        boxes_3d = box_np_ops.center_to_corner_box3d(gt_boxes_camera[:, :3],
                                                     gt_boxes_camera[:, 3:6],
                                                     gt_boxes_camera[:, 6])
        boxes_3d = boxes_3d.reshape((-1, 3))

        boxes_3d_p2 = box_np_ops.project_to_image(boxes_3d, P2)
        boxes_3d_p2 = boxes_3d_p2.reshape([-1, 8, 2])

        inside_mask = []
        for i in range(boxes_3d_p2.shape[0]):
            corners = boxes_3d_p2[i]
            inside_xmin = 0 < corners[:, 0]  #&& 0 < corners[:, 1] < imsize[1])
            inside_xmax = corners[:, 0] < imsize[1]
            inside_x = inside_xmin & inside_xmax

            inside_ymin = 0 < corners[:, 1]  # && 0 < corners[:, 1] < imsize[1])
            inside_ymax = corners[:, 1] < imsize[0]
            inside_y = inside_ymin & inside_ymax
            inside = (inside_x & inside_y).any()
            inside_mask.append(inside)

        #print(boxes_3d_p2)
        boxes_3d_p2 = boxes_3d_p2[inside_mask]
        '''
        bbox_crop = (min(imsize[0], bbox_crop[0]),
                     min(imsize[0], bbox_crop[1]),
                     min(imsize[0], bbox_crop[2]),
                     min(imsize[1], bbox_crop[3]))
        
        # Detect if a cropped box is empty.
        if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
            return None
        '''
        if self.current_image is not None:
            bbox_plot.draw_3d_bbox_in_ax(
                self.w_plt.ax, boxes_3d_p2, colors='b')



    def save_pointcloud(self, idx):
        point_color = self.w_config.get("PointColor")[:3]
        point_color = (*point_color, self.w_config.get("PointAlpha"))
        point_color = np.tile(np.array(point_color), [self.points.shape[0], 1])
        point_size = np.full([self.points.shape[0]], self.w_config.get("PointSize"), dtype=np.float32)
        if 'annos' in self.info:
            gt_point_table = box_np_ops.points_in_rbbox(self.points, self.gt_boxes)
            self.gt_filled_mask = gt_point_table.sum(axis=0)
            gt_point_mask = gt_point_table.any(1)
            point_size[gt_point_mask] = self.w_config.get("GTPointSize")
            gt_point_color = self.w_config.get("GTPointColor")
            gt_point_color = (*gt_point_color[:3], self.w_config.get("GTPointAlpha"))
            point_color[gt_point_mask] = gt_point_color

        range = np.array([self.w_config.get("CoorsRange")])
        #range = np.zeros(range.shape, dtype=range.dtype)
        #range[0] = [-30, -38.4, -10.5, 51.92, 38.4, 9.5]
        bbox = box_np_ops.minmax_to_corner_3d(range)
        self.w_pc_viewer.boxes3d("bound", bbox, GLColor.Green)

        self.w_pc_viewer.remove("dt_boxes/labels")
        self.w_pc_viewer.remove("dt_boxes")
        if self.detection_annos is not None and self.w_config.get("DrawDTBoxes"):
            detection_anno = self.detection_annos[idx]
            self.draw_detection(detection_anno)
        self.w_pc_viewer.scatter(
            "pointcloud", self.points[:, :3], point_color, size=point_size)

    def closeEvent(self, event):
        config_str = self.w_config.dumps()
        self.json_setting.set("config", config_str)
        self.json_setting.set("idx", self.w_idx.text())
        return super().closeEvent(event)

    def on_configchanged(self, msg):
        idx = int(self.w_idx.text())
        config_str = self.w_config.dumps()
        self.json_setting.set("config", config_str)
        pc_redraw_msgs = ["PointSize", "PointAlpha", "GTPointSize"]
        pc_redraw_msgs += ["GTPointAlpha", "WithReflectivity"]
        pc_redraw_msgs += ["PointColor", "GTPointColor"]
        box_redraw = ["GTBoxColor", "GTBoxAlpha"]
        dt_redraw = ["DTBoxColor", "DTBoxAlpha", "DrawDTLabels", "DTScoreAsAlpha", "DTScoreThreshold", "DTBoxLineWidth"]

        vx_redraw_msgs = ["DrawPositiveVoxelsOnly", "DrawVoxels"]
        vx_redraw_msgs += ["PosVoxelColor", "PosVoxelAlpha"]
        vx_redraw_msgs += ["NegVoxelColor", "NegVoxelAlpha"]
        all_redraw_msgs = ["RemoveOutsidePoint"]
        if msg.name in vx_redraw_msgs:
            if self.w_config.get("DrawVoxels"):
                self.w_pc_viewer.draw_voxels(self.points, self.gt_boxes)
            else:
                self.w_pc_viewer.remove("voxels")
        elif msg.name in pc_redraw_msgs:
            self.plot_pointcloud()
        elif msg.name in all_redraw_msgs:
            self.on_plotButtonPressed()
        elif msg.name in box_redraw:
            self.plot_gt_boxes_in_pointcloud()
        elif msg.name in dt_redraw:
            if self.detection_annos is not None and self.w_config.get("DrawDTBoxes"):
                detection_anno = self.detection_annos[idx]
                self.draw_detection(detection_anno)

    def on_loadVxNetCkptPressed(self):
        ckpt_path = Path(self.w_vckpt_path.text())
        self.json_setting.set("latest_vxnet_ckpt_path",
                              self.w_vckpt_path.text())
        self.inference_ctx.restore(ckpt_path)
        # self.w_load_ckpt.setText(self.w_load_ckpt.text() + f": {ckpt_path.stem}")
        self.log("load VoxelNet ckpt succeed.")

    def on_BuildVxNetPressed(self):
        self.inference_ctx = TorchInferenceContext()
        vconfig_path = Path(self.w_vconfig_path.text())
        self.inference_ctx.build(vconfig_path)
        self.json_setting.set("latest_vxnet_cfg_path", str(vconfig_path))
        self.log("Build VoxelNet ckpt succeed.")
        # self.w_load_config.setText(self.w_load_config.text() + f": {vconfig_path.stem}")

    def on_InferenceVxNetPressed(self):
        t = time.time()
        inputs = self.inference_ctx.get_inference_input_dict(
            self.info, self.points)
        self.log("input preparation time:", time.time() - t)
        t = time.time()
        with self.inference_ctx.ctx():
            det_annos = self.inference_ctx.inference(inputs)
        self.log("detection time:", time.time() - t)
        self.draw_detection(det_annos[0])

    def on_LoadInferenceVxNetPressed(self):
        self.on_BuildVxNetPressed()
        self.on_loadVxNetCkptPressed()
        self.on_InferenceVxNetPressed()

    def on_EvalVxNetPressed(self):
        if "annos" not in self.kitti_infos[0]:
            self.error("ERROR: infos don't contain gt label.")
        t = time.time()
        det_annos = []
        input_cfg = self.inference_ctx.config.eval_input_reader
        model_cfg = self.inference_ctx.config.model.second

        class_names = list(input_cfg.class_names)
        num_features = model_cfg.num_point_features
        with self.inference_ctx.ctx():
            for info in list_bar(self.kitti_infos):
                v_path = self.root_path / info['velodyne_path']
                # v_path = v_path.parent.parent / (
                #     v_path.parent.stem + "_reduced") / v_path.name
                points = np.fromfile(
                    str(v_path), dtype=np.float32,
                    count=-1).reshape([-1, num_features])
                rect = info['calib/R0_rect']
                P2 = info['calib/P2']
                Trv2c = info['calib/Tr_velo_to_cam']
                image_shape = info['img_shape']
                if self.w_config.get("RemoveOutsidePoint"):
                    points = box_np_ops.remove_outside_points(
                        points, rect, Trv2c, P2, image_shape)
                inputs = self.inference_ctx.get_inference_input_dict(
                    info, points)
                det_annos += self.inference_ctx.inference(inputs)
        self.log("total detection time:", time.time() - t)
        gt_annos = [i["annos"] for i in self.kitti_infos]
        self.log(get_official_eval_result(gt_annos, det_annos, class_names))

    @staticmethod
    def get_simpify_labels(labels):
        label_map = {
            "Car": "V",
            "Pedestrian": "P",
            "Cyclist": "C",
            "car": "C",
            "tractor": "T1",
            "trailer": "T2",
        }
        label_count = {
            "Car": 0,
            "Pedestrian": 0,
            "Cyclist": 0,
            "car": 0,
            "tractor": 0,
            "trailer": 0,
        }
        ret = []
        for i, name in enumerate(labels):
            count = 0
            if name in label_count:
                count = label_count[name]
                label_count[name] += 1
            else:
                label_count[name] = 0
            ret.append(f"{label_map[name]}{count}")
        return ret

    @staticmethod
    def get_false_pos_neg(gt_boxes, dt_boxes, labels, fp_thresh=0.1):
        iou = riou3d_shapely(gt_boxes, dt_boxes)
        ret = np.full([len(gt_boxes)], 2, dtype=np.int64)
        assigned_dt = np.zeros([len(dt_boxes)], dtype=np.bool_)
        label_thresh_map = {
            "Car": 0.7,
            "Pedestrian": 0.5,
            "Cyclist": 0.5,
            "car": 0.7,
            "tractor": 0.7,
            "trailer": 0.7,
        }
        tp_thresh = np.array([label_thresh_map[n] for n in labels])
        if len(gt_boxes) != 0 and len(dt_boxes) != 0:
            iou_max_dt_for_gt = iou.max(1)
            dt_iou_max_dt_for_gt = iou.argmax(1)
            ret[iou_max_dt_for_gt >= tp_thresh] = 0
            ret[np.logical_and(iou_max_dt_for_gt < tp_thresh,
                               iou_max_dt_for_gt > fp_thresh)] = 1  # FP
            assigned_dt_inds = dt_iou_max_dt_for_gt
            assigned_dt_inds = assigned_dt_inds[iou_max_dt_for_gt >= fp_thresh]
            assigned_dt[assigned_dt_inds] = True
        return ret, assigned_dt


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PCViewer()
    sys.exit(app.exec_())
