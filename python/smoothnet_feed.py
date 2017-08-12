import argparse
import logging
import cv2
import mxnet as mx
from smooth_segnet import get_smooth_segnet
from imageflowiter1 import ImageFlowIter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
FLAGS = None


def train():
    batch_size = FLAGS.batch_size
    stride = FLAGS.stride
    frame_rate = FLAGS.frame_rate
    data_dir = FLAGS.data_dir
    test_dir = FLAGS.test_dir
    image_shape = FLAGS.image_shape
    flow_shape = FLAGS.flow_shape
    label_shape = FLAGS.label_shape
    epoch = FLAGS.epoch
    learning_rate = FLAGS.learning_rate

    if FLAGS.device == 'gpu0':
        context = mx.gpu(0)
    elif FLAGS.device == 'gpu1':
        context = mx.gpu(1)
    elif FLAGS.device == 'gpu01':
        context = [mx.gpu(0), mx.gpu(1)]
    else:
        context = mx.cpu()

    train_iter = ImageFlowIter(data_names=['data', 'flow'],
                                data_shapes=[image_shape, flow_shape],
                                label_names=['softmax_label'],
                                label_shapes=[label_shape],
                                batch_size=batch_size,
                                stride = stride,
                                path_root=data_dir)

    test_iter = ImageFlowIter(data_names=['data', 'flow'],
                               data_shapes=[image_shape, flow_shape],
                               label_names=['softmax_label'],
                               label_shapes=[label_shape],
                               batch_size=batch_size,
                               stride=stride,
                               path_root=test_dir)

    images = mx.sym.var('data')
    flows = mx.sym.var('flow')
    labels = mx.sym.var('softmax_label')

    batch_image_shape = (batch_size,) + image_shape
    batch_flow_shape = (batch_size, frame_rate) + flow_shape
    smooth_net = get_smooth_segnet(images, batch_image_shape, flows, batch_flow_shape, labels)

    smoothnet_model = mx.mod.Module(symbol=smooth_net,
                                    context=context,
                                    data_names=['data', 'flow'])

    # allocate memory given the input data and label shapes
    smoothnet_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    # initialize parameters by uniform random numbers
    smoothnet_model.init_params(initializer=mx.init.Uniform(scale=.07))
    # use SGD with learning rate 0.1 to train
    #lr_scheduler = mx.lr_scheduler.FactorScheduler(step=100, factor=0.9)
    optimizer_params = (('learning_rate', learning_rate), ('momentum', 0.9), ('wd', 0.0005))
    smoothnet_model.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params)
    # use accuracy as the metric
    metric = mx.metric.create('acc')
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(epoch):
        train_iter.reset()
        metric.reset()
        for i, batch in enumerate(train_iter):
            cur_batch_size = batch.data[0].shape[0]
            #print('batch index %d, size: %d' % (i, cur_batch_size))

            smoothnet_model.forward(batch, is_train=True)  # compute predictions
            smoothnet_model.update_metric(metric, batch.label)  # accumulate prediction accuracy
            #print('current label shape: ' + str(batch.label[0].shape))
            smoothnet_model.backward()  # compute gradients
            smoothnet_model.update()  # update parameters
            #print('batch %d training completed' % i)

            if (i > 0) & (i % 10 == 0):
                #print('just for testing............')
                #score = smoothnet_model.score(test_iter, ['acc'], num_batch=10)
                #print('test ended.............')
                print('batch %d, Training %s' % (i, metric.get()))

        print('epoch testing starts ............')
        score = smoothnet_model.score(test_iter, ['acc'])
        print('testing ended.............')
        print('Epoch %d, Training %s, Testing %s \n' % (epoch, metric.get(), score))

    # smoothnet_model.fit(train_iter,
    #                     num_epoch=epoch,
    #                     eval_data=train_iter,  # validation data
    #                     optimizer='sgd',  # use SGD to train
    #                     optimizer_params={'learning_rate': learning_rate},  # use fixed learning rate
    #                     eval_metric='acc',  # report accuracy during training
    #                     batch_end_callback=mx.callback.Speedometer(batch_size, 3))


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
        default='gpu0',
        help="gpu0, gpu1, gpu01 or cpu"
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
        '--stride',
        type=int,
        default=2,
        help='Stride.'
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
        '--test_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the test data.'
    )
    parser.add_argument(
        '--image_shape',
        type=tuple,
        default=(3, 360, 480),
        help='Image shape.'
    )
    parser.add_argument(
        '--flow_shape',
        type=tuple,
        default=(360, 480, 2),
        help='Flow shape.'
    )
    parser.add_argument(
        '--label_shape',
        type=tuple,
        default=(360, 480),
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
