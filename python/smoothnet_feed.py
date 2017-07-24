import argparse
import mxnet as mx
from smoothnet import get_smooth_net
from imageflowiter import ImageFlowIter

FLAGS = None

def train():
    batch_size = FLAGS.batch_size
    context = mx.GPU() if FLAGS.device ==  'GPU' else mx.cpu()
    data_dir = FLAGS.data_dir
    epoch = FLAGS.epoch
    learning_rate = FLAGS.learning_rate

    train_iter = ImageFlowIter(batch_size=batch_size, path_root=data_dir + '/train')

    data = mx.sym.var('data')
    images = data[0]
    flows = data[1]
    labels = mx.sym.var('softmax_label')
    smooth_net = get_smooth_net(images, flows, lables)

    smoothnet_model = mx.mod.Module(symbol=smooth_net, context=context)
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
        default='inference',
        help="Actions: 'train' or 'inference'."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        help="GPU or CPU"
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size.'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data',
        help='Directory of the data.'
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
