"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx

class AssignRoisOperator(mx.operator.CustomOp):
    def __init__(self):
        super(AssignRoisOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0]

        rois_area = mx.ndarray.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))

        aligned_output = mx.ndarray.empty((rois.shape[0], in_data[1].shape[1], 14, 14))
        
        area_threshold = {'P5': 448, 'P4': 224, 'P3': 112}
        rois_p = [
            area_threshold['P5'] <= rois_area,
            mx.ndarray.logical_and(area_threshold['P4'] <= rois_area, rois_area < area_threshold['P5']),
            mx.ndarray.logical_and(area_threshold['P3'] <= rois_area, rois_area < area_threshold['P4']),
            mx.ndarray.logical_and(0 < rois_area, rois_area < area_threshold['P3'])
        ]

        for i in range(4):
            aligned_output[rois_p[i]] = mx.ndarray.contrib.ROIAlign(
                        data=in_data[i+1], rois=rois[rois_p[i]],
                        pooled_size=(14, 14),
                        spatial_scale=1.0 / [32, 16, 8, 4][i])

        self.assign(out_data[0], req[0], aligned_output)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)
        self.assign(in_grad[4], req[4], 0)


@mx.operator.register('assign_rois')
class AssignRoisProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AssignRoisProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['rois', 'P5', 'P4', 'P3', 'P2']

    def list_outputs(self):
        return ['roi_aligned']

    def infer_shape(self, in_shape):

        rois_shape = in_shape[0]
        P2_shape = in_shape[1]
        roi_aligned_shape = (rois_shape[0], P2_shape[1], 14, 14)
        return [rois_shape, in_shape[1], in_shape[2], in_shape[3], in_shape[4]], \
               [roi_aligned_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return AssignRoisOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
