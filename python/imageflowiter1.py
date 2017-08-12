import numpy as np
import mxnet as mx
import os
import random


class ImageFlowIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, label_names, label_shapes, path_root, stride, batch_size=3, frame_rate=30,
                 shuffle=True):
        self._data_names = data_names
        self._data_shapes = data_shapes
        self._label_names = label_names
        self._label_shapes = label_shapes

        self._batch_size = batch_size
        self._stride = stride
        self._path_root = path_root
        self._frame_rate = frame_rate
        self._shuffle = shuffle

        self._subdir_files_dict = {}
        self._dir_file_pairs = self._get_dir_file_pairs()

        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        if self._shuffle:
            random.shuffle(self._dir_file_pairs)

        self._cur_pair_idx = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        image_shape = self._data_shapes[0]
        flow_shape = self._data_shapes[1]
        batch_image_shape = (self._batch_size,) + image_shape
        batch_flow_shape = (self._batch_size, self._frame_rate) + flow_shape

        provide_data = zip(self._data_names, [batch_image_shape, batch_flow_shape])
        return provide_data

    @property
    def provide_label(self):
        batch_label_shape = (self._batch_size,) + self._label_shapes[0]
        provide_label = zip(self._label_names, [batch_label_shape])
        return provide_label

    def next(self):
        if self._cur_pair_idx < len(self._dir_file_pairs):
            dir_file_pair = self._dir_file_pairs[self._cur_pair_idx]
            dir_name = dir_file_pair[0]
            batch_start_idx = dir_file_pair[1]
            batch_end_idx = batch_start_idx + self._batch_size

            # print('A new batch starts:')
            # print('dir:' + dir_name)
            # print('start index:' + str(batch_start_idx))
            # print('end index:' + str(batch_end_idx))

            data = self._read_batch_data(dir_name, batch_start_idx, batch_end_idx)
            labels = self._read_batch_labels(dir_name, batch_start_idx, batch_end_idx)

            self._cur_pair_idx += 1
            return mx.io.DataBatch(data, labels)
        else:
            raise StopIteration

    def _get_dir_file_pairs(self):
        subdirs = [name for name in os.listdir(self._path_root) if os.path.isdir(os.path.join(self._path_root, name))]

        dir_file_pairs = []

        for subdir in subdirs:
            subdir_files = self._get_subdir_files(subdir)

            batch_start_idx = 0
            batch_end_idx = self._batch_size
            while batch_end_idx < len(subdir_files):
                dir_file_pairs.append((subdir, batch_start_idx))
                # print('(%s, %d) added to pairs' % (subdir, batch_start_idx))
                batch_start_idx = batch_start_idx + self._stride
                batch_end_idx = batch_start_idx + self._batch_size

        print('dir file pairs init! There are %d batches!' % len(dir_file_pairs))

        return dir_file_pairs

    def _read_batch_data(self, subdir_name, batch_start_idx, batch_end_idx):
        batch_images = []
        batch_flows = []
        image_dir = self._path_root + "/" + subdir_name + '/' + 'images'
        flow_dir = self._path_root + "/" + subdir_name + '/' + 'flows'

        subdir_files = self._get_subdir_files(subdir_name)
        for i, file_name in enumerate(subdir_files[batch_start_idx:batch_end_idx]):
            image_path = image_dir + '/' + file_name + '.png'
            img = np.transpose(mx.image.imdecode(open(image_path).read()).asnumpy(), (2, 0, 1))
            # print('image file:' + file_name + '; shape:' + str(img.shape))
            batch_images.append(img)

            img_flows = self._get_img_flows(flow_dir, file_name, i < (self._batch_size - 1))
            batch_flows.append(img_flows)

        return [mx.nd.array(batch_images), mx.nd.array(np.array(batch_flows))]

    def _read_batch_labels(self, subdir_name, batch_start_idx, batch_end_idx):
        labels = []
        label_dir = self._path_root + "/" + subdir_name + '/' + 'annot'

        subdir_files = self._get_subdir_files(subdir_name)
        for file_name in subdir_files[batch_start_idx:batch_end_idx]:
            label_path = label_dir + '/' + file_name + '.png'
            label = mx.image.imdecode(open(label_path).read(), flag=0).asnumpy()[:,:,0]
            # print('label shape:' + str(label.shape))
            labels.append(label)

        return [mx.nd.array(labels)]

    def _get_subdir_files(self, subdir_name):
        if subdir_name in self._subdir_files_dict:
            files = self._subdir_files_dict[subdir_name]
        else:
            files = []
            with open(self._path_root + "/" + subdir_name + '/' + 'files.txt', 'r') as f:
                for line in f:
                    files.append(line.strip())
            self._subdir_files_dict[subdir_name] = files

        return files

    def _get_img_flows(self, flow_dir, image_file_name, last_batch_img):
        flow_start_index = int(image_file_name)

        flows = []
        for i in range(self._frame_rate):
            if not last_batch_img:
                flow_path = flow_dir + '/' + str(flow_start_index + i) + '_' + str(flow_start_index + i + 1) + '.flo'
                flow = self._read_flow(flow_path)
            else:
                flow_shape = self._data_shapes[1]
                flow = np.zeros(flow_shape)

            # print('flow shape:' + str(flow.shape))
            flows.append(flow)

        img_flows = np.array(flows)
        return img_flows

    def _read_flow(self, flow_path):
        f = open(flow_path, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
        return flow.astype(np.float32)
