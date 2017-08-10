import numpy as np
import mxnet as mx
import os
import cv2
import random
import math


class ImageIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, label_names, label_shapes, data_root, label_root, imagelist_path, batch_size,
                 shuffle=True):
        self._data_names = data_names
        self._data_shapes = data_shapes
        self._label_names = label_names
        self._label_shapes = label_shapes

        self._batch_size = batch_size
        self._data_root = data_root
        self._label_root = label_root
        self._imagelist_path = imagelist_path

        self._shuffle = shuffle

        self._files = self._get_files()

        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        if self._shuffle:
            random.shuffle(self._files)

        self._cur_batch_size = self._batch_size
        self._cur_batch_idx = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        image_shape = self._data_shapes[0]
        batch_image_shape = (self._cur_batch_size,) + image_shape

        provide_data = zip(self._data_names, [batch_image_shape])
        return provide_data

    @property
    def provide_label(self):
        batch_label_shape = (self._cur_batch_size,) + self._label_shapes[0]
        provide_label = zip(self._label_names, [batch_label_shape])
        return provide_label

    def next(self):
        batch_start_idx = self._cur_batch_idx * self._batch_size
        batch_end_idx = batch_start_idx + self._batch_size

        #print('batch start index:' + str(batch_start_idx))
        #print('batch end index:' + str(batch_end_idx))
        #print('files length:' + str(len(self._files)))

        if (batch_start_idx < len(self._files)) & (batch_end_idx < len(self._files)):
            #print('A new batch starts:')
            batch_start_idx = self._cur_batch_idx * self._batch_size
            batch_end_idx = self._get_batch_end_idx(batch_start_idx)
            #print('start index:' + str(batch_start_idx))
            #print('end index:' + str(batch_end_idx))
            self._cur_batch_size = batch_end_idx - batch_start_idx

            data = self._read_batch_data(batch_start_idx, batch_end_idx)
            labels = self._read_batch_labels(batch_start_idx, batch_end_idx)
            #print("images shape:" + str(data[0].shape))
            #print("labels shape:" + str(labels[0].shape))

            self._cur_batch_idx += 1
            #print('\n')
            return mx.io.DataBatch(data, labels)
        else:
            print('Epoch process completed!')
            raise StopIteration
            #self.reset()
            #return self.next()

    def _get_files(self):
        files = []
        with open(self._imagelist_path, 'r') as f:
            for line in f:
                files.append(line.strip().split())
        return files

    def _read_batch_data(self, batch_start_idx, batch_end_idx):
        batch_images = []

        for i, file_names in enumerate(self._files[batch_start_idx:batch_end_idx]):
            image_file = file_names[0]
            image_path = self._data_root + '/' + image_file
            #print('image path:' + image_path)
            image = cv2.imread(image_path).astype(float)
            #print('shape1:' + str(image.shape))
            img = np.transpose(image, (2, 0, 1))
            #print('shape:' + str(img))
            batch_images.append(img)

        return [mx.nd.array(batch_images)]

    def _read_batch_labels(self, batch_start_idx, batch_end_idx):
        labels = []
        for file_names in self._files[batch_start_idx:batch_end_idx]:
            label_file = file_names[1]
            #print('label file:' + label_file)
            label_path = self._label_root + '/' + label_file
            label = cv2.imread(label_path, 0).astype(float)

            #print('shape:' + str(label))
            labels.append(label)

        return [mx.nd.array(labels)]

    def _get_batch_end_idx(self, batch_start_idx):
        if (batch_start_idx + self._batch_size) < len(self._files):
            return batch_start_idx + self._batch_size
        else:
            return len(self._files)
