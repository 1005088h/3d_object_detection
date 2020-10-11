import control_panel as panel
import bbox_plot
from bbox_plot import GLColor
from framework import box_np_ops
import numpy as np


from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QSizePolicy

from glwidget import KittiGLViewWidget
class KittiPointCloudView(KittiGLViewWidget):
    def __init__(self,
                 config,
                 parent=None,
                 voxel_size=None,
                 coors_range=None,
                 max_voxels=50000,
                 max_num_points=35):
        super().__init__(parent=parent)
        if voxel_size is None:
            voxel_size = [0.2, 0.2, 0.4]
        if coors_range is None:
            coors_range = [0, -40, -3, 70.4, 40, 1]
        self.w_config = config
        self._voxel_size = voxel_size
        self._coors_range = coors_range
        self._max_voxels = max_voxels
        self._max_num_points = max_num_points
        bk_color = (0.8, 0.8, 0.8, 1.0)
        bk_color = list([int(v * 255) for v in bk_color])
        # self.setBackgroundColor(*bk_color)
        # self.w_gl_widget.setBackgroundColor('w')
        self.mousePressed.connect(self.on_mousePressed)
        self.setCameraPosition(distance=20, azimuth=-180, elevation=30)

    def on_mousePressed(self, pos):
        pass

    def reset_camera(self):
        self.set_camera_position(
            center=(5, 0, 0), distance=20, azimuth=-180, elevation=30)
        self.update()

    def draw_frustum(self, bboxes, rect, Trv2c, P2):
        # Y = C(R @ (rect @ Trv2c @ X) + T)
        # uv = [Y0/Y2, Y1/Y2]
        frustums = []
        C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
        frustums = box_np_ops.get_frustum_v2(bboxes, C)
        frustums -= T
        # frustums = np.linalg.inv(R) @ frustums.T
        frustums = np.einsum('ij, akj->aki', np.linalg.inv(R), frustums)
        frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
        self.boxes3d('frustums', frustums, colors=GLColor.Write, alpha=0.5)

    def draw_cropped_frustum(self, bboxes, rect, Trv2c, P2):
        # Y = C(R @ (rect @ Trv2c @ X) + T)
        # uv = [Y0/Y2, Y1/Y2]
        self.boxes3d(
            'cropped_frustums',
            prep.random_crop_frustum(bboxes, rect, Trv2c, P2),
            colors=GLColor.Write,
            alpha=0.5)

    def draw_anchors(self,
                     gt_boxes_lidar,
                     points=None,
                     image_idx=0,
                     gt_names=None):
        # print(gt_names)
        voxel_size = np.array(self._voxel_size, dtype=np.float32)
        # voxel_size = np.array([0.2, 0.2, 0.4], dtype=np.float32)
        coors_range = np.array(self._coors_range, dtype=np.float32)
        # coors_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
        grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        # print(grid_size)
        bv_range = coors_range[[0, 1, 3, 4]]
        anchor_generator = AnchorGeneratorStride(
            # sizes=[0.6, 0.8, 1.73, 0.6, 1.76, 1.73],
            sizes=[0.6, 1.76, 1.73],
            anchor_strides=[0.4, 0.4, 0.0],
            anchor_offsets=[0.2, -39.8, -1.465],
            rotations=[0, 1.5707963267948966],
            match_threshold=0.5,
            unmatch_threshold=0.35,
        )
        anchor_generator1 = AnchorGeneratorStride(
            # sizes=[0.6, 0.8, 1.73, 0.6, 1.76, 1.73],
            sizes=[0.6, 0.8, 1.73],
            anchor_strides=[0.4, 0.4, 0.0],
            anchor_offsets=[0.2, -39.8, -1.465],
            rotations=[0, 1.5707963267948966],
            match_threshold=0.5,
            unmatch_threshold=0.35,
        )
        anchor_generator2 = AnchorGeneratorStride(
            # sizes=[0.6, 0.8, 1.73, 0.6, 1.76, 1.73],
            sizes=[1.6, 3.9, 1.56],
            anchor_strides=[0.4, 0.4, 0.0],
            anchor_offsets=[0.2, -39.8, -1.55442884],
            rotations=[0, 1.5707963267948966],
            # rotations=[0],
            match_threshold=0.6,
            unmatch_threshold=0.45,
        )
        anchor_generators = [anchor_generator2]
        box_coder = GroundBox3dCoder()
        # similarity_calc = DistanceSimilarity(1.0)
        similarity_calc = NearestIouSimilarity()
        target_assigner = TargetAssigner(box_coder, anchor_generators,
                                         similarity_calc)
        # anchors = box_np_ops.create_anchors_v2(
        #     bv_range, grid_size[:2] // 2, sizes=anchor_dims)
        # matched_thresholds = [0.45, 0.45, 0.6]
        # unmatched_thresholds = [0.3, 0.3, 0.45]

        t = time.time()
        feature_map_size = grid_size[:2] // 2
        feature_map_size = [*feature_map_size, 1][::-1]
        print(feature_map_size)
        # """
        ret = target_assigner.generate_anchors(feature_map_size)
        anchors = ret["anchors"]
        anchors = anchors.reshape([-1, 7])
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])
        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]
        print(f"num_anchors_ {len(anchors)}")
        if points is not None:
            voxels, coors, num_points = points_to_voxel(
                points,
                self._voxel_size,
                # self._coors_range,
                coors_range,
                self._max_num_points,
                reverse_index=True,
                max_voxels=self._max_voxels)

            # print(np.min(coors, 0), np.max(coors, 0))
            dense_voxel_map = box_np_ops.sparse_sum_for_anchors_mask(
                coors, tuple(grid_size[::-1][1:]))
            dense_voxel_map = dense_voxel_map.cumsum(0)
            dense_voxel_map = dense_voxel_map.cumsum(1)
            anchors_mask = box_np_ops.fused_get_anchors_area(
                dense_voxel_map, anchors_bv, voxel_size, coors_range,
                grid_size) > 1
        print(np.sum(anchors_mask), anchors_mask.shape)
        class_names = [
            'Car', "Pedestrian", "Cyclist", 'Van', 'Truck', "Tram", 'Misc',
            'Person_sitting'
        ]
        gt_classes = np.array(
            [class_names.index(n) + 1 for n in gt_names], dtype=np.int32)
        t = time.time()
        target_dict = target_assigner.assign(
            anchors,
            gt_boxes_lidar,
            anchors_mask,
            gt_classes=gt_classes,
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds)
        labels = target_dict["labels"]
        reg_targets = target_dict["bbox_targets"]
        reg_weights = target_dict["bbox_outside_weights"]
        # print(labels[labels > 0])
        # decoded_reg_targets = box_np_ops.second_box_decode(reg_targets, anchors)
        # print(decoded_reg_targets.reshape(-1, 7)[labels > 0])
        print("target time", (time.time() - t))
        print(f"num_pos={np.sum(labels > 0)}")
        colors = np.zeros([anchors.shape[0], 4])
        ignored_color = bbox_plot.gl_color(GLColor.Gray, 0.5)
        pos_color = bbox_plot.gl_color(GLColor.Cyan, 0.5)

        colors[labels == -1] = ignored_color
        colors[labels > 0] = pos_color
        cared_anchors_mask = np.logical_and(labels != 0, anchors_mask)
        colors = colors[cared_anchors_mask]
        anchors_not_neg = box_np_ops.rbbox3d_to_corners(anchors)[
            cared_anchors_mask]
        self.boxes3d("anchors", anchors_not_neg, colors=colors)


    def draw_bounding_box(self, CoorsRange):
        bbox = box_np_ops.minmax_to_corner_3d(np.array([CoorsRange]))
        self.boxes3d("bound", bbox, GLColor.Green)

    def draw_voxels(self, points, gt_boxes=None):
        pos_color = self.w_config.get("PosVoxelColor")[:3]
        pos_color = (*pos_color, self.w_config.get("PosVoxelAlpha"))
        neg_color = self.w_config.get("NegVoxelColor")[:3]
        neg_color = (*neg_color, self.w_config.get("NegVoxelAlpha"))

        voxel_size = np.array(self.w_config.get("VoxelSize"), dtype=np.float32)
        coors_range = np.array(
            self.w_config.get("CoorsRange"), dtype=np.float32)
        voxels, coors, num_points = points_to_voxel(
            points,
            voxel_size,
            coors_range,
            self._max_num_points,
            reverse_index=True,
            max_voxels=self._max_voxels)
        # print("num_voxels", num_points.shape[0])
        """
        total_num_points = 0
        for i in range(self._max_num_points):
            num = np.sum(num_points.astype(np.int64) == i)
            total_num_points += num * i
            if num > 0:
                print(f"num={i} have {num} voxels")
        print("total_num_points", points.shape[0], total_num_points)
        """
        grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        shift = coors_range[:3]
        voxel_origins = coors[:, ::-1] * voxel_size + shift
        voxel_maxs = voxel_origins + voxel_size
        voxel_boxes = np.concatenate([voxel_origins, voxel_maxs], axis=1)
        voxel_box_corners = box_np_ops.minmax_to_corner_3d(voxel_boxes)
        pos_only = self.w_config.get("DrawPositiveVoxelsOnly")
        if gt_boxes is not None:
            labels = box_np_ops.assign_label_to_voxel(
                gt_boxes, coors, voxel_size, coors_range).astype(np.bool)
            if pos_only:
                voxel_box_corners = voxel_box_corners[labels]
            colors = np.zeros([voxel_box_corners.shape[0], 4])
            if pos_only:
                colors[:] = pos_color
            else:
                colors[np.logical_not(labels)] = neg_color
                colors[labels] = pos_color
        else:
            if not pos_only:
                colors = np.zeros([voxel_box_corners.shape[0], 4])
                colors[:] = neg_color
            else:
                voxel_box_corners = np.zeros((0, 8, 3))
                colors = np.zeros((0, 4))
        self.boxes3d("voxels", voxel_box_corners, colors)


class MatPlotLibView(FigureCanvas):
    def __init__(self, parent=None, rect=[5, 4], dpi=100):
        # super().__init__()
        self.fig = Figure(figsize=(rect[0], rect[1]), dpi=dpi)
        self.ax = self.fig.add_subplot(1, 1, 1)
        # self.ax.axis('off')
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        #self.axes.set_ylim([-1,1])
        #self.axes.set_xlim([0,31.4159*2])
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.draw()

    def reset_plot(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(1, 1, 1)

class KittiDrawControl(panel.ControlPanel):
    def __init__(self, title, parent=None):
        super().__init__(column_nums=[2, 1, 1, 2], tab_num=1, parent=parent)
        self.setWindowTitle(title)
        with self.tab(0, "common"):
            with self.column(0):
                self.add_listedit("UsedClass", str)
                self.add_fspinbox("PointSize", 0.01, 0.5, 0.01, 0.05)
                self.add_fspinbox("PointAlpha", 0.0, 1.0, 0.05, 0.5)
                self.add_colorbutton("PointColor",
                                     bbox_plot.gl_color(GLColor.Gray))
                self.add_fspinbox("GTPointSize", 0.01, 0.5, 0.01, 0.2)
                self.add_fspinbox("GTPointAlpha", 0.0, 1.0, 0.05, 0.5)
                self.add_colorbutton("GTPointColor",
                                     bbox_plot.gl_color(GLColor.Purple))
                self.add_checkbox("WithReflectivity")
                self.add_checkbox("DrawGTBoxes")
                self.add_checkbox("DrawGTLabels")
                self.add_colorbutton("GTBoxColor",
                                     bbox_plot.gl_color(GLColor.Green))
                self.add_fspinbox("GTBoxAlpha", 0.0, 1.0, 0.05, 0.5)
                self.add_checkbox("DrawDTBoxes")

                self.add_checkbox("DrawDTLabels")
                self.add_checkbox("DTScoreAsAlpha")
                self.add_fspinbox("DTScoreThreshold", 0.0, 1.0, 0.01, 0.3)
                self.add_colorbutton("DTBoxColor",
                                     bbox_plot.gl_color(GLColor.Blue))
                self.add_fspinbox("DTBoxAlpha", 0.0, 1.0, 0.05, 0.5)
                self.add_fspinbox("DTBoxLineWidth", 0.25, 10.0, 0.25, 1.0)
            with self.column(1):
                self.add_arrayedit("CoorsRange", np.float64,
                                   [-40, -40, -2, 40, 40, 4], [6])
                self.add_arrayedit("VoxelSize", np.float64, [0.2, 0.2, 0.4],
                                   [3])
                self.add_checkbox("DrawVoxels")
                self.add_colorbutton("PosVoxelColor",
                                     bbox_plot.gl_color(GLColor.Yellow))
                self.add_fspinbox("PosVoxelAlpha", 0.0, 1.0, 0.05, 0.5)
                self.add_colorbutton("NegVoxelColor",
                                     bbox_plot.gl_color(GLColor.Purple))
                self.add_fspinbox("NegVoxelAlpha", 0.0, 1.0, 0.05, 0.5)
                self.add_checkbox("DrawPositiveVoxelsOnly")
                self.add_checkbox("RemoveOutsidePoint")
