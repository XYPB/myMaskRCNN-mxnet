import mxnet as mx
import numpy as np

from symdata.anchor import AnchorGenerator, AnchorSampler
from symdata.image import imdecode, resize, transform, get_image, tensor_vstack

def load_test(filename, short, max_size, mean, std):
    # read and transform image
    im_orig = imdecode(filename)
    im, im_scale = resize(im_orig, short, max_size)
    height, width = im.shape[:2]
    im_info = mx.nd.array([height, width, im_scale])

    # transform into tensor and normalize
    im_tensor = transform(im, mean, std)

    # for 1-batch inference purpose, cannot use batchify (or nd.stack) to expand dims
    im_tensor = mx.nd.array(im_tensor).expand_dims(0)
    im_info = mx.nd.array(im_info).expand_dims(0)

    # transform cv2 BRG image to RGB for matplotlib
    im_orig = im_orig[:, :, (2, 1, 0)]
    return im_tensor, im_info, im_orig


def generate_batch(im_tensor, im_info):
    """return batch"""
    data = [im_tensor, im_info]
    data_shapes = [('data', im_tensor.shape), ('im_info', im_info.shape)]
    data_batch = mx.io.DataBatch(data=data, label=None, provide_data=data_shapes, provide_label=None)
    return data_batch


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size, short, max_size, mean, std):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self._roidb = roidb
        self._batch_size = batch_size
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std

        # infer properties from roidb
        self._size = len(self._roidb)
        self._index = np.arange(self._size)

        # decide data and label names (only for training)
        self._data_name = ['data', 'im_info']
        self._label_name = None

        # status variable
        self._cur = 0
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return None

    def reset(self):
        self._cur = 0

    def iter_next(self):
        return self._cur + self._batch_size <= self._size

    def next(self):
        if self.iter_next():
            data_batch = mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self._cur += self._batch_size
            return data_batch
        else:
            raise StopIteration

    def getdata(self):
        indices = self.getindex()
        im_tensor, im_info = [], []
        for index in indices:
            roi_rec = self._roidb[index]
            b_im_tensor, b_im_info, _ = get_image(roi_rec, self._short, self._max_size, self._mean, self._std)
            im_tensor.append(b_im_tensor)
            im_info.append(b_im_info)
        im_tensor = mx.nd.array(tensor_vstack(im_tensor, pad=0))
        im_info = mx.nd.array(tensor_vstack(im_info, pad=0))
        self._data = im_tensor, im_info
        return self._data

    def getlabel(self):
        return None

    def getindex(self):
        cur_from = self._cur
        cur_to = min(cur_from + self._batch_size, self._size)
        return np.arange(cur_from, cur_to)

    def getpad(self):
        return max(self._cur + self.batch_size - self._size, 0)


class AnchorLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size, short, max_size, mean, std,
                 feat_sym, anchor_generator: AnchorGenerator, anchor_sampler: AnchorSampler,
                 shuffle=False):
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self._roidb = roidb
        self._batch_size = batch_size
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._feat_sym = feat_sym
        self._ag = anchor_generator
        self._as = anchor_sampler
        self._shuffle = shuffle

        # infer properties from roidb
        self._size = len(roidb)
        self._index = np.arange(self._size)

        # decide data and label names
        self._data_name = ['data', 'im_info', 'gt_boxes']
        self._label_name = ['label',
                            'bbox_target',
                            'bbox_weight',
                            ]

        # status variable
        self._cur = 0
        self._data = None
        self._label = None

        # get first batch to fill in provide_data and provide_label
        self.next()
        self.reset()

    @property
    def provide_data(self):
        return [(k, v.shape) for k, v in zip(self._data_name, self._data)]

    @property
    def provide_label(self):
        return [(k, v.shape) for k, v in zip(self._label_name, self._label)]

    def reset(self):
        self._cur = 0
        if self._shuffle:
            np.random.shuffle(self._index)

    def iter_next(self):
        return self._cur + self._batch_size <= self._size

    def next(self):
        if self.iter_next():
            data_batch = mx.io.DataBatch(data=self.getdata(), label=self.getlabel(),
                                         pad=self.getpad(), index=self.getindex(),
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            self._cur += self._batch_size
            return data_batch
        else:
            raise StopIteration

    def getdata(self):
        indices = self.getindex()
        im_tensor, im_info, gt_boxes = [], [], []
        for index in indices:
            roi_rec = self._roidb[index]
            b_im_tensor, b_im_info, b_gt_boxes = get_image(roi_rec, np.random.randint(640, 800), self._max_size, self._mean, self._std)
            im_tensor.append(b_im_tensor)
            im_info.append(b_im_info)
            gt_boxes.append(b_gt_boxes)
        im_tensor = mx.nd.array(tensor_vstack(im_tensor, pad=0))
        im_info = mx.nd.array(tensor_vstack(im_info, pad=0))
        gt_boxes = mx.nd.array(tensor_vstack(gt_boxes, pad=-1))
        self._data = im_tensor, im_info, gt_boxes
        return self._data

    def getlabel(self):
        im_tensor, im_info, gt_boxes = self._data

        label_list = []
        bbox_target_list = []
        bbox_weight_list = []
        num_anchors = 3
        # all stacked image share same anchors
        # print(type(self._feat_sym))
        for i, feat_sym in enumerate(self._feat_sym):
            _, out_shape, _ = feat_sym.infer_shape(data=im_tensor.shape)
            feat_height, feat_width = out_shape[0][-2:]
            # print('feat height & width', feat_height, feat_width)
            anchors = self._ag[i].generate(feat_height, feat_width)

        # assign anchor according to their real size encoded in im_info
            label, bbox_target, bbox_weight = [], [], []
            for batch_ind in range(im_info.shape[0]):
                b_im_info = im_info[batch_ind].asnumpy()
                b_gt_boxes = gt_boxes[batch_ind].asnumpy()
                b_im_height, b_im_width = b_im_info[:2]

                b_label, b_bbox_target, b_bbox_weight = self._as.assign(anchors, b_gt_boxes, b_im_height, b_im_width)

                b_label = b_label.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1)).flatten()
                b_bbox_target = b_bbox_target.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))
                b_bbox_weight = b_bbox_weight.reshape((feat_height, feat_width, -1)).transpose((2, 0, 1))

                label.append(b_label)
                bbox_target.append(b_bbox_target)
                bbox_weight.append(b_bbox_weight)

            label_list.append(mx.nd.array(tensor_vstack(label, pad=-1)))
            bbox_target_list.append(mx.nd.array(tensor_vstack(bbox_target, pad=0)))
            bbox_weight_list.append(mx.nd.array(tensor_vstack(bbox_weight, pad=0)))
        label_list = mx.nd.concat(*[i.reshape((0,-1)) for i in label_list], dim=1)
        bbox_target_list = mx.nd.concat(*[i.reshape((0,12,-1)) for i in bbox_target_list], dim=2)
        bbox_weight_list = mx.nd.concat(*[i.reshape((0,12,-1)) for i in bbox_weight_list], dim=2)
        self._label = label_list, bbox_target_list, bbox_weight_list
        return self._label

    def getindex(self):
        cur_from = self._cur
        cur_to = min(cur_from + self._batch_size, self._size)
        return np.arange(cur_from, cur_to)

    def getpad(self):
        return max(self._cur + self.batch_size - self._size, 0)
