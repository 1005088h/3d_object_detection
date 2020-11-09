import torch
import numpy as np
from enum import Enum
from torch import nn


class LossGenerator:

    def __init__(self, config):
        super().__init__()
        self._pos_cls_weight = 1.0
        self._neg_cls_weight = 1.0
        self._loss_norm_type = LossNormType.NormByNumPositives
        self._box_code_size = config['box_code_size']
        self._num_class = 1
        self.loc_loss_type = WeightedSmoothL1LocalizationLoss()
        self.cls_loss_type = SigmoidFocalClassificationLoss()
        self.dir_loss_type = WeightedSoftmaxClassificationLoss()
        self._loc_loss_weight = 2.0
        self._cls_loss_weight = 1.0
        self._direction_loss_weight = 0.2
        self._use_direction_classifier = True
        self.device = config['device']

    def generate(self, preds_dict, example):
        # prepare loss inputs
        labels = torch.from_numpy(example['labels']).to(self.device)
        reg_targets = torch.from_numpy(example['bbox_targets']).to(self.device)

        # normalization weights
        cls_weights, reg_weights, cared = self.prepare_loss_weights(labels, dtype=reg_targets.dtype)
        cls_targets = labels * cared.type_as(labels)
        cls_preds = preds_dict['cls_preds']
        box_preds = preds_dict['box_preds']

        # create losses
        batch_size = int(box_preds.shape[0])
        box_preds = box_preds.view(batch_size, -1, self._box_code_size)
        cls_preds = cls_preds.view(batch_size, -1, self._num_class)
        one_hot_targets = one_hot(cls_targets, depth=2, dtype=box_preds.dtype)
        one_hot_targets = one_hot_targets[..., 1:]
        # sin(a - b) = sinacosb-cosasinb

        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
        loc_loss = self.loc_loss_type.compute_loss(box_preds, reg_targets, weights=reg_weights)
        cls_loss = self.cls_loss_type.compute_loss(cls_preds, one_hot_targets, weights=cls_weights)

        # compile losses
        batch_size = cls_loss.shape[0]
        loc_loss_reduced = loc_loss.sum() / batch_size * self._loc_loss_weight
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_loss_reduced = cls_loss.sum() / batch_size * self._cls_loss_weight
        loss = loc_loss_reduced + cls_loss_reduced

        dir_cls_targets = torch.from_numpy(example['dir_cls_targets']).cuda()
        dir_cls_targets = one_hot(dir_cls_targets, 2)
        dir_logits = preds_dict["dir_cls_preds"].view(batch_size, -1, 2)
        weights = (labels > 0).type_as(dir_logits)
        weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
        dir_loss = self.dir_loss_type.compute_loss(dir_logits, dir_cls_targets, weights=weights)
        dir_loss = dir_loss.sum() / batch_size
        loss += dir_loss * self._direction_loss_weight

        return {
            "loss": loss,
            "cls_pos_loss": cls_pos_loss,
            "cls_neg_loss": cls_neg_loss,
            "dir_loss": dir_loss,
            "cls_loss": cls_loss_reduced,
            "loc_loss": loc_loss_reduced
        }

    def prepare_loss_weights(self, labels, dtype=torch.float32):
        """get cls_weights and reg_weights from labels.
        """
        cared = labels >= 0
        # cared: [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = self._neg_cls_weight * negatives.type(dtype)
        positive_cls_weights = self._pos_cls_weight * positives.type(dtype)
        cls_weights = negative_cls_weights + positive_cls_weights
        reg_weights = positives.type(dtype)
        if self._loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
        elif self._loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        elif self._loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer
        else:
            raise ValueError(f"unknown loss norm type.")
        return cls_weights, reg_weights, cared


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    # sin(a - b) = sinacosb-cosasinb
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


class SigmoidFocalClassificationLoss:
    """Sigmoid focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """

    def __init__(self, gamma=2.0, alpha=0.25):
        """Constructor.

        Args:
          gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
          alpha: optional alpha weighting factor to balance positives vs negatives.
          all_zero_negative: bool. if True, will treat all zero as background.
            else, will treat first label as background. only affect alpha.
        """
        self._alpha = alpha
        self._gamma = gamma

    def compute_loss(self, prediction_tensor, target_tensor, weights):
        weights = weights.unsqueeze(2)
        per_entry_cross_ent = (_sigmoid_cross_entropy_with_logits(labels=target_tensor, logits=prediction_tensor))
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha + (1 - target_tensor) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor * per_entry_cross_ent)
        return focal_cross_entropy_loss * weights


def _sigmoid_cross_entropy_with_logits(logits, labels):
    loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))

    return loss


class WeightedSmoothL1LocalizationLoss:
    """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

    def __init__(self, sigma=3.0):
        super().__init__()
        self._sigma = sigma
        code_weights = np.ones(7, dtype=np.float32)
        self._code_weights = torch.from_numpy(code_weights).cuda()

    def compute_loss(self, prediction_tensor, target_tensor, weights):
        diff = prediction_tensor - target_tensor
        code_weights = self._code_weights.type_as(prediction_tensor)
        diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma ** 2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) + (abs_diff - 0.5 / (self._sigma ** 2)) * (
                1. - abs_diff_lt_1)
        anchorwise_smooth_l1norm = loss * weights.unsqueeze(-1)
        return anchorwise_smooth_l1norm


class WeightedSoftmaxClassificationLoss:
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        self._logit_scale = logit_scale

    def compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(
            prediction_tensor, self._logit_scale)
        per_row_cross_ent = (_softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)))
        return per_row_cross_ent.view(weights.shape) * weights


def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param)  # [N, ..., C] -> [N, C, ...]
    #loss_ftor = nn.CrossEntropyLoss(reduce=False)
    loss_ftor = nn.CrossEntropyLoss(reduction='none')
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss
