import numpy as np
import mxnet as mx
from smoothop import Smooth


def convolution(name, data, data_shape, num_filter, kernel=(7, 7), stride=(1, 1), pad=(3, 3), act_type="relu",
                batch_norm=True, workspace=2048):
    output = mx.sym.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter,
                                workspace=workspace)

    if batch_norm:
        output = mx.sym.BatchNorm(output)
    if act_type != None:
        output = mx.sym.Activation(output, act_type=act_type)

    _, out_shapes, _ = output.infer_shape(data=data_shape)
    print("Layer " + name + " output shape: " + str(out_shapes))

    return output


def upsampling(name, data, data_shape, num_filter, scale=2, sample_type="bilinear", act_type="relu", batch_norm=True, workspace=2048):
    output = mx.symbol.UpSampling(data, scale=scale, sample_type=sample_type, workspace=workspace, num_filter=num_filter)

    if batch_norm:
        output = mx.sym.BatchNorm(output)
    if act_type != None:
        output = mx.sym.Activation(output, act_type=act_type)

    _, out_shapes, _ = output.infer_shape(data=data_shape)
    print("Layer " + name + " output shape: " + str(out_shapes))

    return output


def pooling(name, data, data_shape, pool_type="max", kernel=(2, 2), stride=(2, 2)):
    output = mx.sym.Pooling(data, pool_type=pool_type, kernel=kernel, stride=stride)

    _, out_shapes, _ = output.infer_shape(data=data_shape)
    print("Layer " + name + " output shape: " + str(out_shapes))

    return output


def get_smooth_segnet(images, images_shape, flows, flows_shape, labels):
    num_filter = 64

    conv1 = convolution('conv1', images, images_shape, num_filter)
    pool1 = pooling('pool1', conv1, images_shape)

    conv2 = convolution('conv2', pool1, images_shape, num_filter)
    pool2 = pooling('pool2', conv2, images_shape)

    conv3 = convolution('conv3', pool2, images_shape, num_filter)
    pool3 = pooling('pool3', conv3, images_shape)

    conv4 = convolution('conv4', pool3, images_shape, num_filter)
    pool4 = pooling('pool4', conv4, images_shape)

    upsampling4 = upsampling('upsampling4', pool4, images_shape, num_filter)
    conv_decode4 = convolution('conv_decode4', upsampling4, images_shape, num_filter, kernel=(6, 7), act_type=None)

    upsampling3 = upsampling('upsampling3', conv_decode4, images_shape, num_filter)
    conv_decode3 = convolution('conv_decode3', upsampling3, images_shape, num_filter, act_type=None)

    upsampling2 = upsampling('upsampling2', conv_decode3, images_shape, num_filter)
    conv_decode2 = convolution('conv_decode2', upsampling2, images_shape, num_filter, act_type=None)

    upsampling1 = upsampling('upsampling1', conv_decode2, images_shape, num_filter)
    conv_decode1 = convolution('conv_decode1', upsampling1, images_shape, num_filter, act_type=None)

    smooth_conv_decode1 = mx.symbol.Custom(conv_decode1, flows, op_type='smooth')
    _, out_shapes, _ = smooth_conv_decode1.infer_shape(data=images_shape, flow=flows_shape)
    print("Layer smooth_conv_decode1 output shape: " + str(out_shapes))

    conv_classifier = mx.sym.Convolution(data=smooth_conv_decode1, num_filter=11, kernel=(1, 1), pad=(0, 0),
                                         stride=(1, 1), workspace=2048)
    _, out_shapes, _ = conv_classifier.infer_shape(data=images_shape, flow=flows_shape)
    print("Layer conv_classifier output shape: " + str(out_shapes))

    smooth_segnet = mx.sym.SoftmaxOutput(data=conv_classifier, label=labels, ignore_label=11, multi_output=True, use_ignore=True)
    _, out_shapes, _ = smooth_segnet.infer_shape(data=images_shape, flow=flows_shape)
    print("Net output shape: " + str(out_shapes) + '\n')

    return smooth_segnet
