"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np


class OutputDebugOperator(mx.operator.CustomOp):
    def __init__(self):
        super(OutputDebugOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        all_vals = in_data[0].asnumpy()
        print(all_vals)
        self.assign(out_data[0], req[0], all_vals)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], in_data[0])


@mx.operator.register('output_debug')
class OutputDebugProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(OutputDebugProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['vals']

    def list_outputs(self):
        return ['vals']

    def infer_shape(self, in_shape):
        rpn_vals_shape = in_shape[0]

        output_vals_shape = in_shape[0]

        return [rpn_vals_shape], \
               [output_vals_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return OutputDebugOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
