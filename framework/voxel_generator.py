import numba
import numpy as np



class VoxelGenerator:
    def __init__(self, config):

        detection_range = np.array(config['detection_range'], dtype=np.float32)
        detection_center = (detection_range[3:] + detection_range[:3]) / 2
        voxel_size = np.array(config['voxel_size'], dtype=np.float32)
        range = detection_range[3:] - detection_range[:3]
        grid_size = range / voxel_size
        grid_size = grid_size.astype(np.int32)
        range_diff = grid_size.astype(voxel_size.dtype) * voxel_size
        offset = detection_center - range_diff / 2
        detection_range = np.concatenate((offset, offset + range_diff), axis=0)

        self.voxel_size = voxel_size
        self.detection_range = detection_range
        self.offset = offset
        self.grid_size = grid_size
        self.max_num_points = config['max_num_points']
        self.max_voxels = config['max_voxels']
        config['detection_range'] = detection_range
        config['detection_offset'] = offset
        config['detection_range_diff'] = range_diff
        config['grid_size'] = grid_size

    def generate(self, points):
        voxels = np.zeros(shape=(self.max_voxels, self.max_num_points, points.shape[-1]), dtype=points.dtype)
        num_points_per_voxel = np.zeros(shape=(self.max_voxels,), dtype=np.int32)
        coors = np.zeros(shape=(self.max_voxels, 3), dtype=np.int32)
        coor_to_voxelidx = -np.ones(shape=self.grid_size, dtype=np.int32)

        voxel_num = points_to_voxels(points, voxels, num_points_per_voxel, coors, coor_to_voxelidx, self.voxel_size,
                                    self.offset, self.grid_size, self.max_voxels, self.max_num_points)
        coors = coors[:voxel_num]
        voxels = voxels[:voxel_num]
        num_points_per_voxel = num_points_per_voxel[:voxel_num]

        return voxels, coors, num_points_per_voxel


@numba.jit(nopython=True)
def points_to_voxels(points, voxels, num_points_per_voxel, coors, coor_to_voxelidx, voxel_size,
                     offset, grid_size, max_voxels, max_num_points):
    voxel_num = 0
    num_points = points.shape[0]
    coor = np.zeros(shape=(3,), dtype=np.int32)
    for i in range(num_points):
        for d in range(3):
            coor[d] = np.floor((points[i][d] - offset[d]) / voxel_size[d])
        inside = (coor >= 0).all() and (coor < grid_size).all()
        if not inside:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            if voxel_num >= max_voxels:
                break
            voxelidx = voxel_num
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
            voxel_num += 1
        num = num_points_per_voxel[voxelidx]
        if num < max_num_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num
