import argparse
import logging
import cv2
import mxnet as mx
from smoothnet import get_smooth_net
from imageflowiter import ImageFlowIter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
FLAGS = None


def train():
    batch_size = FLAGS.batch_size
    context = mx.gpu() if FLAGS.device == 'gpu' else mx.cpu()
    frame_rate = FLAGS.frame_rate
    data_dir = FLAGS.data_dir
    image_shape = FLAGS.image_shape
    flow_shape = FLAGS.flow_shape
    label_shape = FLAGS.label_shape
    epoch = FLAGS.epoch
    learning_rate = FLAGS.learning_rate

    train_iter = ImageFlowIter(data_names=['data', 'flow'],
                               data_shapes=[image_shape, flow_shape],
                               label_names=['softmax_label'],
                               label_shapes=[label_shape],
                               batch_size=batch_size,
                               path_root=data_dir + '/train')

    images = mx.sym.var('data')
    flows = mx.sym.var('flow')
    labels = mx.sym.var('softmax_label')

    batch_image_shape = (batch_size,) + image_shape
    batch_flow_shape = (batch_size, frame_rate) + flow_shape
    smooth_net = get_smooth_net(images, batch_image_shape, flows, batch_flow_shape, labels)

    smoothnet_model = mx.mod.Module(symbol=smooth_net,
                                    context=context,
                                    data_names=['data', 'flow'])

    smoothnet_model.fit(train_iter,
                        num_epoch=epoch,
                        optimizer_params={'learning_rate':learning_rate})


def inference():
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--action',
        type=str,
        default='train',
        help="Actions: 'train' or 'inference'."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='gpu',
        help="gpu or cpu"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=5,
        help='Batch size.'
    )
    parser.add_argument(
        '--frame_rate',
        type=int,
        default=30,
        help='Frame rate in the video.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the data.'
    )
    parser.add_argument(
        '--image_shape',
        type=tuple,
        default=(3, 720, 960),
        help='Image shape.'
    )
    parser.add_argument(
        '--flow_shape',
        type=tuple,
        default=(720, 960, 2),
        help='Flow shape.'
    )
    parser.add_argument(
        '--label_shape',
        type=tuple,
        default=(720, 960),
        help='Label shape.'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default='10',
        help='Train epoch.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/data',
        help='Directory to put the log data, including trained model data.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.action == 'train':
        train()
    elif FLAGS.action == 'inference':
        inference()
