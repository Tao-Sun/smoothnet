import numpy as np 
import mxnet as mx
from smoothnet import Smooth

def convolution(data, num_filter, kernel=(7, 7), stride=(1, 1), pad=(3, 3), act_type="relu", batch_norm=True, workspace=2048):
    output = mx.sym.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter, workspace=workspace)
    
    if batch_norm:
        output = mx.sym.BatchNorm(output)
    if act_type != None:
        output = mx.sym.Activation(output, act_type=act_type)

    return output

def deconv(data, num_filter, kernel=(2, 2), pad=(0, 0), stride=(2, 2), act_type="relu", batch_norm=True, workspace=2048):
    output = mx.sym.Deconvolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter, workspace=workspace)
    
    if batch_norm:
        output = mx.sym.BatchNorm(output)
    if act_type != None:
        output = mx.sym.Activation(output, act_type=act_type)

    return output

def pooling(data, pool_type="max", kernel=(2, 2), stride=(2, 2)):
    output = mx.sym.Pooling(data, pool_type=pool_type, kernel=kernel, stride=stride)
    return output

def net(images, flows, labels):
    num_filter = 64

    conv1 = convolution(images, num_filter=num_filter)
    pool1 = pooling(conv1)

    conv2 = convolution(pool1, num_filter=num_filter)
    pool2 = pooling(conv2)

    conv3 = convolution(pool2, num_filter=num_filter)
    pool3 = pooling(conv3)

    conv4 = convolution(pool3, num_filter=num_filter)
    pool4 = pooling(conv4)

    deconv4 = deconv(pool4, num_filter=num_filter)
    conv_decode4 = convolution(deconv4, num_filter=num_filter, act_type=None)

    deconv3 = deconv(conv_decode4, num_filter=num_filter)
    conv_decode3 = convolution(deconv3, num_filter=num_filter, act_type=None)

    deconv2 = deconv(conv_decode3, num_filter=num_filter)
    conv_decode2 = convolution(deconv2, num_filter=num_filter, act_type=None)

    deconv1 = deconv(conv_decode2, num_filter=num_filter)
    conv_decode1 = convolution(deconv1, num_filter=num_filter, act_type=None)

    warped_conv_decode1 = Smooth([conv_decode1, flows])
    combined_conv_decode1 = conv_decode1 + warped_conv_decode1

    conv_classifier = convolution(combined_conv_decode1, num_filter=11, kernel=(1, 1), pad=(0, 0), batch_norm=False, act_type=None)

    smoothnet = mx.sym.softmaxoutput(data=conv_classifier, label=labels, multi_output=True, ignore_label=11, use_ignore=True)
    return smoothnet


