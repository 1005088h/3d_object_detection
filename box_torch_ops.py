import numpy as np
import torch
from torch import stack as tstack


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners


def corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1).astype(np.float32)
    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=np.float32)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2 ** ndim, ndim)
    return corners


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def rotation_2d(points, angles):
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    rot_mat_T = torch.stack(
        [tstack([rot_cos, rot_sin]),
         tstack([-rot_sin, rot_cos])])
    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def box_decode(box_encodings, anchors):
    xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, lt, wt, ht, rt = torch.split(box_encodings, 1, dim=-1)
    za = za + ha / 2

    diagonal = torch.sqrt(la ** 2 + wa ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * ha + za

    lg = torch.exp(lt) * la
    wg = torch.exp(wt) * wa
    hg = torch.exp(ht) * ha

    rg = rt + ra
    zg = zg - hg / 2
    return torch.cat([xg, yg, zg, lg, wg, hg, rg], dim=-1)

