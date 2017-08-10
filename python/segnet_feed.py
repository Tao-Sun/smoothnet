import argparse
import logging
import cv2
import mxnet as mx
from segnet import get_segnet
from imageiter import ImageIter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
FLAGS = None


def train():
    batch_size = FLAGS.batch_size
    data_dir = FLAGS.data_dir
    label_dir = FLAGS.label_dir
    imagelist_path = FLAGS.imagelist_path
    image_shape = FLAGS.image_shape
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

    train_iter = ImageIter(data_names=['data'],
                           data_shapes=[image_shape],
                           label_names=['softmax_label'],
                           label_shapes=[label_shape],
                           data_root=data_dir,
                           label_root=label_dir,
                           imagelist_path=imagelist_path,
                           batch_size=batch_size)

    images = mx.sym.var('data')
    labels = mx.sym.var('softmax_label')

    batch_image_shape = (batch_size,) + image_shape
    segnet = get_segnet(images, batch_image_shape, labels)

    segnet_model = mx.mod.Module(symbol=segnet,
                                 context=context,
                                 data_names=['data'])

    # allocate memory given the input data and label shapes
    segnet_model.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)

    # initialize parameters by uniform random numbers
    segnet_model.init_params(initializer=mx.init.Uniform(scale=.07))
    # use SGD with learning rate 0.1 to train
    #lr_scheduler = mx.lr_scheduler.FactorScheduler(step=100, factor=0.9)
    optimizer_params = (('learning_rate', learning_rate), ('momentum', 0.9), ('wd', 0.0005))
    segnet_model.init_optimizer(optimizer='sgd', optimizer_params=optimizer_params)
    # use accuracy as the metric
    metric = mx.metric.create('acc')
    # train 5 epochs, i.e. going over the data iter one pass
    for epoch in range(epoch):
        train_iter.reset()
        metric.reset()
        for i, batch in enumerate(train_iter):
            cur_batch_size = batch.data[0].shape[0]
            #print('current batch %d, size: %d' % (i, cur_batch_size))
            binded_batch_size = segnet_model.data_shapes[0].shape[0]
            if cur_batch_size != binded_batch_size:
                print('previous batch size: %d ;rebinding to batch size: %d' % (binded_batch_size, cur_batch_size))
                segnet_model.bind(data_shapes=train_iter.provide_data,
                                  label_shapes=train_iter.provide_label,
                                  force_rebind=True)

            segnet_model.forward(batch, is_train=True)  # compute predictions
            #print(batch.label[0][0].asnumpy())
            segnet_model.update_metric(metric, batch.label)  # accumulate prediction accuracy
            #print('hererree5')
            #print('current label shape: ' + str(batch.label[0].shape))
            segnet_model.backward()  # compute gradients
            #print('hererree3')
            segnet_model.update()  # update parameters
            #print('hererree4')

            if (i - 1) % 3 == 0:
                #print('hererree1')
                print('batch %d, Training %s' % (i, metric.get()))
            #print('hererree2')
        print('Epoch %d, Training %s' % (epoch, metric.get()))


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
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the data.'
    )
    parser.add_argument(
        '--label_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the labels. '
    )
    parser.add_argument(
        '--imagelist_path',
        type=str,
        default='/tmp/data',
        help='Path of the image list file. '
    )
    parser.add_argument(
        '--image_shape',
        type=tuple,
        default=(3, 360, 480),
        help='Image shape.'
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
