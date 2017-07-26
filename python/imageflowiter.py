import numpy as np
import mxnet as mx
import os
import random
import cv2

class ImageFlowIter(mx.io.DataIter):
    def __init__(self, data_names, data_shapes, label_names, label_shapes, path_root, batch_size=3, frame_rate=30, shuffle=True):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self._data_names = data_names
        self._batch_size = batch_size
        self._path_root = path_root
        self._frame_rate = frame_rate
        self._shuffle = shuffle
        
        self._subdirs = [name for name in os.listdir(path_root) if os.path.isdir(os.path.join(path_root, name))]

        self._cur_subdir_idx = 0
        self._cur_subdir_files = self._get_cur_subdir_files()
        self._subdir_batch_num = self._get_subdir_batch_num()
        self._cur_batch_idx = 0

    def __iter__(self):
        return self

    def reset(self):
        if self._shuffle:
            random.shuffle(self._subdir)

        self._cur_subdir_idx = 0
        self._cur_subdir_files = self._get_cur_subdir_files()
        self._subdir_batch_num = 0
        self._cur_batch_idx = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self._cur_subdir_idx < len(self._subdirs):
            if self._cur_batch_idx < self._subdir_batch_num:
                data = self._read_batch_data()
                labels = self._read_batch_labels()
                print("images shape:" + str(data[0].shape))
                print("flows shape:" + str(data[1].shape))
                print("labels shape:" + str(labels[0].shape))
                
                self._cur_batch_idx += 1
                return mx.io.DataBatch(data, labels)
            else:
                self._cur_subdir_idx += 1
                if self._cur_subdir_idx < len(self._subdir):
                    self._cur_subdir_files = self._get_cur_subdir_files()
                    self._subdir_batch_num = self._get_subdir_batch_num()
                    self._cur_batch_idx = 0

                    self.next()
        else:
            raise StopIteration

    def _get_cur_subdir_files(self):
        subdir_name = self._subdirs[self._cur_subdir_idx]
        file_pairs = []
        with open(self._path_root + "/" + subdir_name + '/' + 'files.txt', 'r') as f:
            for line in f:
                img_file, label_file = line.split()
                file_pairs.append((img_file, label_file))

        return file_pairs

    def _get_subdir_batch_num(self):
        return len(self._cur_subdir_files) / self._batch_size + 1
    
    def _read_batch_data(self):
        batch_start_idx = self._cur_batch_idx * self._batch_size
        batch_end_idx = self._get_batch_end_idx(batch_start_idx)

        
        batch_images = []
        batch_flows = []
        image_dir = self._path_root + "/" + self._subdirs[self._cur_subdir_idx] + '/' + 'images'
        flow_dir = self._path_root + "/" + self._subdirs[self._cur_subdir_idx] + '/' + 'flows'

        for i, (image_file, _) in enumerate(self._cur_subdir_files[batch_start_idx:batch_end_idx]):
            image_path = image_dir + '/' + self._subdirs[self._cur_subdir_idx] + '_' + image_file
            print('image path:' + image_path)
            img = np.transpose(mx.image.imdecode(open(image_path).read()).asnumpy(), (2, 0, 1))
            batch_images.append(img)

            if i < (batch_end_idx-batch_start_idx):
                img_flows = self._get_img_flows(flow_dir, image_file)
                batch_flows.append(img_flows)
            else:
                batch_flows.append(np.zeros(img_flows.shape))

        return [mx.nd.array(batch_images), mx.nd.array(np.array(batch_flows))]

    def _read_batch_labels(self):
        batch_start_idx = self._cur_batch_idx * self._batch_size
        batch_end_idx = self._get_batch_end_idx(batch_start_idx)

        labels = []
        label_dir = self._path_root + "/" + self._subdirs[self._cur_subdir_idx] + '/' + 'annot'
        for _, label_file in self._cur_subdir_files[batch_start_idx:batch_end_idx]:
            label_path = label_dir + '/' + self._subdirs[self._cur_subdir_idx] + '_' + label_file
            print('label path:' + label_path)
            label = mx.image.imdecode(open(label_path).read(), flag=0).asnumpy()[:,:,0]
            labels.append(label)

        return [mx.nd.array(labels)]

    def _get_batch_end_idx(self, batch_start_idx):
        if (batch_start_idx + self._batch_size) < len(self._cur_subdir_files):
            return (batch_start_idx + self._batch_size)
        else:
            return len(self._cur_subdir_files)

    def _get_img_flows(self, flow_dir, image_file):
        flow_start_index = int(image_file[0:image_file.index('.png')])

        flows = []
        for i in range(self._frame_rate):
            flow_path = flow_dir + '/' +str(flow_start_index + i) + '_' + str(flow_start_index + i + 1) + '.flo'
            flow = self._read_flow(flow_path)
            flows.append(flow)

        img_flows = np.array(flows)
        return img_flows

    def _read_flow(self, file_name):
        f = open(file_name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
        return flow.astype(np.float32)
