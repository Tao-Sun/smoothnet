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
        self._cur_batch_num = self._get_cur_batch_num()
        self._cur_batch_idx = 0

    def __iter__(self):
        return self

    def reset(self):
        if self._shuffle:
            random.shuffle(self._subdir)

        self._cur_subdir_idx = 0
        self._cur_subdir_files = self._get_cur_subdir_files()
        self._cur_batch_num = 0
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
        if self._cur_subdir_idx < len(self._subdir):
            if self._cur_batch_idx < self._cur_batch_num:
                data = mx.nd.array(self._read_batch_data())
                label = mx.nd.array(self._read_batch_label())
                self.cur_batch_idx += 1
                return mx.io.DataBatch(data, label)
            else:
                self._cur_subdir_idx += 1
                if self._cur_subdir_idx < len(self._subdir):
                    self._cur_subdir_files = self._get_cur_subdir_files()
                    self._cur_batch_num = self.sel_get_cur_batch_num()
                    self._cur_batch_idx = 0
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

    def _get_cur_batch_num(self):
        return len(self._cur_subdir_files) / self._batch_size + 1
    
    def _read_batch_data(self):
        batch_start_idx = self._cur_batch_idx * self._batch_size
        batch_end_idx = self._get_batch_end_idx()
        
        batch_images = []
        batch_flows = []
        for i, (image_file, _) in enumerate(self._cur_subdir_files[batch_start_idx:batch_end_idx]):
            image_dir = self._path_root + "/" + self._subdirs[self._cur_subdir_idx] + '/' + 'images' + '/'
            image_path = image_dir + self._subdirs[self._cur_subdir_idx] + '_' + image_file
            print(image_path)
            img = np.swapaxes(cv2.imread(image_path), 0, 2)
            batch_images.append(img)

            if i > 0:
                img_flows = self._get_img_flows(image_file)
                batch_flows.append(img_flows)
            else:
                batch_flows.append(None)

        return [batch_images, batch_flows]

    def _read_batch_label(self):
        batch_start_idx = self._cur_batch_idx * self._batch_size
        batch_end_idx = self._get_batch_end_idx()

        labels = []
        for i, (image_file, _) in enumerate(self._cur_subdir_files[batch_start_idx:batch_end_idx]):
            label_path = self._path_root + "/" + self._subdirs[self._cur_subdir_idx] + '/' + 'annot' + '/' + image_file
            label = cv2.imread(label_path, 0)
            labels.append(label)

        return np.array(labels)

    def _get_batch_end_idx(self):
        if (batch_start_idx + self._batch_size) < len(self._cur_subdir_files) :
            return (batch_start_idx + self._batch_size)
        else:
            return len(self._cur_subdir_files)


    def _get_img_flows(self, image_file):
        flow_start_index = int(image_file[0:image_file.index('.png')]) - self._frame_rate

        img_flows = []
        for i in range(self._frame_rate):
            flow_dir = self._path_root + "/" + self._subdirs[self._cur_subdir_idx] + '/' + 'flows'
            flow_file_name = flow_dir + str(flow_start_index + i) + '_' + str(flow_start_index + i + 1) + '.flo' 
            flow = self._read_flow(flow_file_name)
            
            img_flows.append(flow) 

        return img_flows

    def _read_flow(self, file_name):
        if name.endswith('.pfm') or name.endswith('.PFM'):
            return readPFM(name)[0][:,:,0:2]

        f = open(name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2)) 

        return flow.astype(np.float32)
