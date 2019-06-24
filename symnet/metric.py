import mxnet as mx
import numpy as np

RPN_FEAT_STRIDE = [64, 32, 16, 8, 4]

def get_names():
    pred = ['rpn_cls_output64', 'rpn_cls_output32', 'rpn_cls_output16', 'rpn_cls_output8', 'rpn_cls_output4',
            'rpn_bbox_loss64', 'rpn_bbox_loss32', 'rpn_bbox_loss16', 'rpn_bbox_loss8', 'rpn_bbox_loss4',
            'rcnn_cls_prob', 'rcnn_bbox_loss', 'rcnn_label']
    label = ['label_stride64', 'label_stride32', 'label_stride16', 'label_stride8', 'label_stride4',
            'bbox_target_stride64', 'bbox_target_stride32', 'bbox_target_stride16', 'bbox_target_stride8', 'bbox_target_stride4',
            'bbox_weight_stride64', 'bbox_weight_stride32', 'bbox_weight_stride16', 'bbox_weight_stride8', 'bbox_weight_stride4',]
    return pred, label


class RPNAccMetricS64(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetricS64, self).__init__('RPNAcc_S64')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output64')]
        label = labels[self.label.index('label_stride64')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class RPNAccMetricS32(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetricS32, self).__init__('RPNAcc_S32')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output32')]
        label = labels[self.label.index('label_stride32')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class RPNAccMetricS16(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetricS16, self).__init__('RPNAcc_S16')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output16')]
        label = labels[self.label.index('label_stride16')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class RPNAccMetricS8(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetricS8, self).__init__('RPNAcc_S8')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output8')]
        label = labels[self.label.index('label_stride8')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)

class RPNAccMetricS4(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetricS4, self).__init__('RPNAcc_S4')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output4')]
        label = labels[self.label.index('label_stride4')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)






class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)






class RPNLogLossMetricS64(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetricS64, self).__init__('RPNLogLoss_S64')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output64')]
        label = labels[self.label.index('label_stride64')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RPNLogLossMetricS32(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetricS32, self).__init__('RPNLogLoss_S32')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output32')]
        label = labels[self.label.index('label_stride32')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RPNLogLossMetricS16(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetricS16, self).__init__('RPNLogLoss_S16')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output16')]
        label = labels[self.label.index('label_stride16')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RPNLogLossMetricS8(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetricS8, self).__init__('RPNLogLoss_S8')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output8')]
        label = labels[self.label.index('label_stride8')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]

class RPNLogLossMetricS4(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetricS4, self).__init__('RPNLogLoss_S4')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_output4')]
        label = labels[self.label.index('label_stride4')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]







class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        label = preds[self.pred.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')
        # print(pred.shape)
        # print(label.shape[0])
        cls = pred[np.arange(label.shape[0]), label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]







class RPNL1LossMetricS64(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetricS64, self).__init__('RPNL1Loss_S64')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss64')].asnumpy()
        bbox_weight = labels[self.label.index('bbox_weight_stride64')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class RPNL1LossMetricS32(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetricS32, self).__init__('RPNL1Loss_S32')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss32')].asnumpy()
        bbox_weight = labels[self.label.index('bbox_weight_stride32')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class RPNL1LossMetricS16(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetricS16, self).__init__('RPNL1Loss_S16')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss16')].asnumpy()
        bbox_weight = labels[self.label.index('bbox_weight_stride16')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class RPNL1LossMetricS8(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetricS8, self).__init__('RPNL1Loss_S8')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss8')].asnumpy()
        bbox_weight = labels[self.label.index('bbox_weight_stride8')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst

class RPNL1LossMetricS4(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetricS4, self).__init__('RPNL1Loss_S4')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss4')].asnumpy()
        bbox_weight = labels[self.label.index('bbox_weight_stride4')].asnumpy()

        # calculate num_inst (average on those fg anchors)
        num_inst = np.sum(bbox_weight > 0) / 4

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst








class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.pred, self.label = get_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        label = preds[self.pred.index('rcnn_label')].asnumpy()

        # calculate num_inst
        keep_inds = np.where(label != 0)[0]
        num_inst = len(keep_inds)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst
