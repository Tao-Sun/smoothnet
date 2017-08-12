import os
import cv2
import mxnet as mx
import numpy as np


class Smooth(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        batch_feature_maps = in_data[0]
        #print('batch_feature_maps shape:' + str(batch_feature_maps))
        batch_series_flows = in_data[1]
        #print('batch_flow shape:' + str(batch_series_flows))

        batch_warped__maps = self._warp_feature_maps(batch_feature_maps, batch_series_flows)
        batch_combined_maps = batch_feature_maps + batch_warped__maps
        self.assign(out_data[0], req[0], batch_combined_maps)
        #self.assign(out_data[0], req[0], mx.nd.array(batch_feature_maps))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        out_maps_grad = out_grad[0]
        #print('out_maps_grad shape:' + str(out_maps_grad.shape))
        in_maps_grad = out_maps_grad.copy()
        in_maps_grad[0] = 2 * in_maps_grad[0]

        self.assign(in_grad[0], req[0], mx.nd.array(in_maps_grad))
        #self.assign(in_grad[0], req[0], mx.nd.array(out_maps_grad))

    def _warp_feature_maps(self, batch_feature_maps, batch_series_flows):
        batch_warped_maps = batch_feature_maps
        batch_series_flows = mx.nd.transpose(batch_series_flows, (1, 0, 4, 2, 3))

        for batch_flows in batch_series_flows:
            batch_warped_maps = self._warp_image(batch_warped_maps, batch_flows)

        return batch_warped_maps

    def _warp_image(self, img, flow):
        flow = mx.nd.transpose(flow, (1, 0, 2, 3))
        h, w = flow.shape[2:4]
        #flow = -flow
        #print('111img shape:' + str(img))
        #print('111flow shape' + str(flow))
        flow[0] += mx.nd.arange(w, ctx=flow[0].context)
        #print('111flow0 shape:' + str(flow[0]))
        flow[1] += mx.nd.reshape(mx.nd.arange(h, ctx=flow[1].context), (h, 1))
        #print('111flow1 shape:' + str(flow[1]))
        flow = mx.nd.transpose(flow, (1, 0, 2, 3))
        #print('111flow shape' + str(flow) + '\n')

        res = mx.nd.BilinearSampler(img, flow)
        #res = img
        #print('result feature shape:' + str(res.shape) + '\n')
        return res

@mx.operator.register("smooth")
class SmoothProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(SmoothProp, self).__init__(need_top_grad=True)
    
    def list_arguments(self):
        return ['data', 'flow']

    def list_outputs(self):
        return ['output']

    # def infer_shape(self, in_shape):
    #     data_shape = in_shape[0]
    #     output_shape = in_shape[0]
    #     return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return Smooth()
