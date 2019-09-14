"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
import cv2
import PIL.Image as Image
import math
import time
import threading as td
from queue import Queue

from symdata.bbox import bbox_overlaps, bbox_transform


def compute_mask_and_label_single_py(roi, label, ins_seg, q, n):
    class_id = [0, 24, 25, 26, 27, 28, 31, 32, 33]
    # print(roi)
    target = ins_seg[int(0.5+roi[1]): int(0.5+roi[3]), int(0.5+roi[0]): int(0.5+roi[2])]
    ids = np.unique(target)
    ins_id = 0
    max_count = 0
    ids = ids[np.floor(ids / 1000) == class_id[int(label)]]
    # ids = ids[(ids // 1000) == label]
    if len(ids) == 1:
        ins_id = ids[0]
        max_count = 1
    else:
        for id in ids:
            px = np.where(ins_seg == int(id))
            x_min = np.min(px[1])
            y_min = np.min(px[0])
            x_max = np.max(px[1])
            y_max = np.max(px[0])
            x1 = max(roi[0], x_min)
            y1 = max(roi[1], y_min)
            x2 = min(roi[2], x_max)
            y2 = min(roi[3], y_max)
            iou = (x2 - x1) * (y2 - y1)
            iou = iou / ((roi[2] - roi[0]) * (roi[3] - roi[1])
                            + (x_max - x_min) * (y_max - y_min) - iou)
                            
            # print(math.floor(id / 1000), x_min, y_min, x_max, y_max, iou)
            if iou > max_count:
                ins_id = id
                max_count = iou

    if max_count == 0:
        q.put((n, np.zeros((28, 28), dtype=np.float32), 0))

    else:
        # print max_count
        mask = np.zeros(target.shape, dtype=np.float32)
        idx = np.where(target == ins_id)
        mask[idx] = 1
        px = idx
        x_min = np.min(px[1])
        y_min = np.min(px[0])
        x_max = np.max(px[1])
        y_max = np.max(px[0])
        if (x_max - x_min) * (y_max - y_min) > 0:

            mask = mask[y_min:y_max, x_min:x_max]
            mask = cv2.resize(mask, (28, 28), interpolation=cv2.INTER_LINEAR)
        # print(mask[0,:].sum(),mask[13,:].sum(),mask[:,0].sum(),mask[:,13].sum())

            q.put((n, mask, label))
        else:
            q.put((n, np.zeros((28, 28), dtype=np.float32), 0))


def compute_mask_and_label(ex_rois, ex_labels, seg):
    # assert os.path.exists(seg_gt), 'Path does not exist: {}'.format(seg_gt)
    # im = Image.open(seg_gt)
    # pixel = list(im.getdata())
    # pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
    # seg = np.int32(seg)
    # print(ins_seg.shape)
    rois = ex_rois[:,1:]
    # print(rois)
    n_rois = ex_rois.shape[0]
    label = ex_labels
    mask_target = np.zeros((n_rois, 28, 28), dtype=np.float32)
    mask_label = np.zeros((n_rois), dtype=np.uint8)
    q = Queue()
    t_list = []
    for n in range(n_rois):
        t = td.Thread(target=compute_mask_and_label_single_py, args=(rois[n], label[n], seg, q, n))
        t.start()
        t_list.append(t)
    for t in t_list:
        t.join()
        n, _mask, _label = q.get()
        # t.kill()
        del t
        mask_target[n] = _mask
        mask_label[n] = _label
    return mask_target, mask_label


def sample_rois(rois, gt_boxes, num_classes, rois_per_image, fg_rois_per_image, fg_overlap, box_stds, seg, im_info):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: [n, 5] (batch_index, x1, y1, x2, y2)
    :param gt_boxes: [n, 5] (x1, y1, x2, y2, cls)
    :param num_classes: number of classes
    :param rois_per_image: total roi number
    :param fg_rois_per_image: foreground roi number
    :param fg_overlap: overlap threshold for fg rois
    :param box_stds: std var of bbox reg
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    overlaps = bbox_overlaps(rois[:, 1:], gt_boxes[:,:4])
    gt_assignment = overlaps.argmax(axis=1)
    labels = gt_boxes[gt_assignment, 4]
    max_overlaps = overlaps.max(axis=1)

    # select foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(max_overlaps >= fg_overlap)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_this_image = min(fg_rois_per_image, len(fg_indexes))
    # sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_this_image:
        fg_indexes = np.random.choice(fg_indexes, size=fg_rois_this_image, replace=False)

    # select background RoIs as those within [0, FG_THRESH)
    bg_indexes = np.where(max_overlaps < fg_overlap)[0]
    # compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_this_image = rois_per_image - fg_rois_this_image
    bg_rois_this_image = min(bg_rois_this_image, len(bg_indexes))
    # sample bg rois without replacement
    if len(bg_indexes) > bg_rois_this_image:
        bg_indexes = np.random.choice(bg_indexes, size=bg_rois_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)
    # pad more bg rois to ensure a fixed minibatch size
    while len(keep_indexes) < rois_per_image:
        gap = min(len(bg_indexes), rois_per_image - len(keep_indexes))
        gap_indexes = np.random.choice(range(len(bg_indexes)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, bg_indexes[gap_indexes])

    # sample rois and labels
    rois = rois[keep_indexes]
    labels = labels[keep_indexes]
    # set labels of bg rois to be 0
    labels[fg_rois_this_image:] = 0

    mask_targets = np.zeros((rois_per_image, num_classes, 28, 28), dtype=np.float32)
    mask_weights = np.zeros((rois_per_image, num_classes, 1, 1), dtype=np.float32)
    # print(im_info[2])
    _mask_targets, _mask_labels = compute_mask_and_label(rois[:fg_rois_this_image], labels[:fg_rois_this_image], seg)
    for i in range(fg_rois_this_image):
        if _mask_labels[i]:
            mask_targets[i, _mask_labels[i]] = _mask_targets[i]
            # print(mask_targets[i, _mask_labels[i]])
            mask_weights[i, _mask_labels[i]] = 1
    # load or compute bbox_target
    targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4], box_stds=box_stds)
    bbox_targets = np.zeros((rois_per_image, 4 * num_classes), dtype=np.float32)
    bbox_weights = np.zeros((rois_per_image, 4 * num_classes), dtype=np.float32)
    for i in range(fg_rois_this_image):
        cls_ind = int(labels[i])
        bbox_targets[i, cls_ind * 4:(cls_ind + 1) * 4] = targets[i]
        bbox_weights[i, cls_ind * 4:(cls_ind + 1) * 4] = 1

    return rois, labels, bbox_targets, bbox_weights, mask_targets, mask_weights


class ProposalTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction, fg_overlap, box_stds):
        super(ProposalTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._rois_per_image = int(batch_rois / batch_images)
        self._fg_rois_per_image = int(round(fg_fraction * self._rois_per_image))
        self._fg_overlap = fg_overlap
        self._box_stds = box_stds

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_images == in_data[1].shape[0], 'check batch size of gt_boxes'

        all_rois = in_data[0].asnumpy()
        all_gt_boxes = in_data[1].asnumpy()
        all_segs = np.uint32(in_data[2].asnumpy()+0.5)
        all_im_info = in_data[3].asnumpy()
        # print(all_segs.shape)

        rois = np.empty((0, 5), dtype=np.float32)
        labels = np.empty((0, ), dtype=np.float32)
        bbox_targets = np.empty((0, 4 * self._num_classes), dtype=np.float32)
        bbox_weights = np.empty((0, 4 * self._num_classes), dtype=np.float32)
        mask_targets = np.empty((0, self._num_classes, 28, 28), dtype=np.int8)
        mask_weights = np.empty((0, self._num_classes, 1, 1), dtype=np.int8)
        for batch_idx in range(self._batch_images):
            b_rois = all_rois[np.where(all_rois[:, 0] == batch_idx)[0]]
            b_gt_boxes = all_gt_boxes[batch_idx]
            b_gt_boxes = b_gt_boxes[np.where(b_gt_boxes[:, -1] > 0)[0]]
            b_segs = all_segs[batch_idx]
            b_im_info = all_im_info[batch_idx]

            # Include ground-truth boxes in the set of candidate rois
            batch_pad = batch_idx * np.ones((b_gt_boxes.shape[0], 1), dtype=b_gt_boxes.dtype)
            b_rois = np.vstack((b_rois, np.hstack((batch_pad, b_gt_boxes[:, :-1]))))

            b_rois, b_labels, b_bbox_targets, b_bbox_weights, b_mask_targets, b_mask_weights = \
                sample_rois(b_rois, b_gt_boxes, num_classes=self._num_classes, rois_per_image=self._rois_per_image,
                            fg_rois_per_image=self._fg_rois_per_image, fg_overlap=self._fg_overlap, box_stds=self._box_stds, seg=b_segs, im_info=b_im_info)

            rois = np.vstack((rois, b_rois))
            labels = np.hstack((labels, b_labels))
            bbox_targets = np.vstack((bbox_targets, b_bbox_targets))
            bbox_weights = np.vstack((bbox_weights, b_bbox_weights))
            mask_targets = np.vstack((mask_targets, b_mask_targets))
            mask_weights = np.vstack((mask_weights, b_mask_weights))

        self.assign(out_data[0], req[0], rois)
        self.assign(out_data[1], req[1], labels)
        self.assign(out_data[2], req[2], bbox_targets)
        self.assign(out_data[3], req[3], bbox_weights)
        self.assign(out_data[4], req[4], mask_targets)
        self.assign(out_data[5], req[5], mask_weights)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register('proposal_target')
class ProposalTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes='21', batch_images='1', batch_rois='128', fg_fraction='0.25',
                 fg_overlap='0.5', box_stds='(0.1, 0.1, 0.2, 0.2)'):
        super(ProposalTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)
        self._fg_overlap = float(fg_overlap)
        self._box_stds = tuple(np.fromstring(box_stds[1:-1], dtype=float, sep=','))

    def list_arguments(self):
        return ['rois', 'gt_boxes', 'seg', 'im_info']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight', 'mask_targets', 'mask_weights']

    def infer_shape(self, in_shape):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)

        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        seg_shape = in_shape[2]
        im_info_shape = in_shape[3]

        output_rois_shape = (self._batch_rois, 5)
        label_shape = (self._batch_rois, )
        bbox_target_shape = (self._batch_rois, self._num_classes * 4)
        bbox_weight_shape = (self._batch_rois, self._num_classes * 4)
        mask_target_shape = (self._batch_rois, self._num_classes, 28, 28)
        mask_weight_shape = (self._batch_rois, self._num_classes, 1, 1)

        return [rpn_rois_shape, gt_boxes_shape, seg_shape, im_info_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape, mask_target_shape, mask_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction,
                                      self._fg_overlap, self._box_stds)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
