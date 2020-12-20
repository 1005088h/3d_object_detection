class MultiHead(nn.Module):

    def __init__(self, in_plane):
        super().__init__()
        self.box_code_size = 7

        num_veh_size = 3
        num_veh_rot = 2
        num_veh_anchor_per_loc = num_veh_size * num_veh_rot
        self.conv_veh_cls = nn.Conv2d(in_plane, num_veh_anchor_per_loc, 1)
        self.conv_veh_box = nn.Conv2d(in_plane, num_veh_anchor_per_loc * self.box_code_size, 1)
        self.conv_veh_dir = nn.Conv2d(in_plane, num_veh_anchor_per_loc * 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_preds = self.conv_veh_cls(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
        box_preds = self.conv_veh_box(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.box_code_size)
        dir_preds = self.conv_veh_dir(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)

        pred_dict = {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_preds": dir_preds
        }

        return pred_dict

