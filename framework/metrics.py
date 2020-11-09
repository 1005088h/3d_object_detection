import torch
import numpy as np


class Metric:
    def __init__(self):
        self._thresholds = [0.1, 0.3, 0.5, 0.7]
        num = len(self._thresholds)
        self.rec_count = torch.zeros(num).cuda()
        self.rec_total = torch.zeros(num).cuda()
        self.prec_count = torch.zeros(num).cuda()
        self.prec_total = torch.zeros(num).cuda()

    def update(self, labels, preds, weights=None):
        with torch.no_grad():
            scores = torch.sigmoid(preds.cuda())
            scores = torch.max(scores, dim=-1)[0]
            labels = torch.from_numpy(labels).cuda()
            if weights is None:
                weights = (labels != -1).float()
            else:
                weights = weights.float()
            for i, thresh in enumerate(self._thresholds):
                tp, tn, fp, fn = _calc_binary_metrics(labels, scores, weights, thresh)
                rec_count = tp + fn
                prec_count = tp + fp
                if rec_count > 0:
                    self.rec_count[i] += rec_count
                    self.rec_total[i] += tp
                if prec_count > 0:
                    self.prec_count[i] += prec_count
                    self.prec_total[i] += tp

    def __str__(self):
        str = ""
        prec, rec = self.value
        for i, t in enumerate(self._thresholds):
            str += "@%.2f prec:%.5f, rec:%.5f  " % (t, prec[i], rec[i])
        return str

    @property
    def value(self):
        prec_count = torch.clamp(self.prec_count, min=1.0)
        rec_count = torch.clamp(self.rec_count, min=1.0)
        return (self.prec_total / prec_count).cpu(), (self.rec_total / rec_count).cpu()

    def clear(self):
        self.rec_count.zero_()
        self.prec_count.zero_()
        self.prec_total.zero_()
        self.rec_total.zero_()


def _calc_binary_metrics(labels, scores, weights=None, threshold=0.5):
    pred_labels = (scores > threshold).long()
    N, *Ds = labels.shape
    labels = labels.view(N, int(np.prod(Ds)))
    pred_labels = pred_labels.view(N, int(np.prod(Ds)))
    pred_trues = pred_labels > 0
    pred_falses = pred_labels == 0
    trues = labels > 0
    falses = labels == 0
    true_positives = (weights * (trues & pred_trues).float()).sum()
    true_negatives = (weights * (falses & pred_falses).float()).sum()
    false_positives = (weights * (falses & pred_trues).float()).sum()
    false_negatives = (weights * (trues & pred_falses).float()).sum()
    return true_positives, true_negatives, false_positives, false_negatives
