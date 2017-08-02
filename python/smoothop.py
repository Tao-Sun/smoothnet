import os
import cv2
import mxnet as mx
import numpy as np


class Smooth(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        batch_feature_maps = in_data[0].asnumpy()
        batch_flow = in_data[1].asnumpy()
        frame_rate = 30

        warped_feature_maps = self._warp_feature_maps(batch_feature_maps, batch_flow, frame_rate)
        combined_feature_maps = batch_feature_maps + warped_feature_maps
        self.assign(out_data[0], req[0], mx.nd.array(combined_feature_maps))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        in_maps_grad = []
        out_maps_grad = out_grad[0].asnumpy()
        for i, out_map_grad in enumerate(out_maps_grad):
            if i == 0:
                in_map_grad = 2 * out_map_grad
            else:
                in_map_grad = out_map_grad

            in_maps_grad.append(in_map_grad)

        self.assign(in_grad[0], req[0], mx.nd.array(np.array(in_maps_grad)))

    def _warp_feature_maps(self, batch_maps, batch_flow):
        batch_warped_maps = [np.transpose(batch_maps[0], (1, 2, 0))]

        for i, feature_map in enumerate(batch_maps[0:batch_maps.shape[0]-1]):
            img_map = np.transpose(feature_map, (1, 2, 0))
            img_flows = batch_flow[i]
            for flow in img_flows:
                img_map = self._warp_image(img_map, flow)

            batch_warped_maps.append(img_map)

        return np.transpose(np.array(batch_warped_maps), (0, 3, 1, 2))

    def _warp_image(self, img, flow):
        h, w = flow.shape[:2]
        #flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:, np.newaxis]
        res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
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
