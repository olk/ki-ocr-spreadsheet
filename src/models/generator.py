import csv
import cv2
import numpy as np

from collections import Counter
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self,
                 dir_p,
                 img_w, img_h,
                 batch_size,
                 downsample_factor):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.max_text_len = 0
        samples = []
        text = ' '
        with open(dir_p.joinpath('labels.csv'), 'rt') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_p = dir_p.joinpath(row['image'])
                # image must exist
                assert img_p.exists()
                label = row['label'].strip()
                # label must not be empty
                assert 0 < len(label)
                samples.append([img_p, label])
                text += label
                if len(label) > self.max_text_len:
                    self.max_text_len = len(label)
        self.alphabet = sorted(list(set(Counter(text).keys())))
        self.n = len(samples)
        # check for images as many entries in labels.csv
        assert len(set(dir_p.glob('*.jpg'))) == self.n
        self.indexes = list(range(self.n))
        self.curr_idx = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.labels = []
        self.sources = []
        for i, (img_p, label) in enumerate(samples):
            label = label.strip()
            img = cv2.imread(str(img_p))
            assert img.shape[0] == self.img_h
            assert img.shape[1] == self.img_w
            # decode JPEG to RGB grid of pixels (gray)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
            # convert to floating-point tensor
            img = img.astype(np.float32)
            # rescale pixel values to the [0,1] interval
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.labels.append(label)
            self.sources.append(img_p.name)
        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(self.n/self.batch_size))

    def __getitem__(self, idx):
        # generate one batch of data
        # generate indexes of the batch
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def __data_generation(self, indexes):
        # generates data containing batch-size samples
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        # initialize input shape
        if 'channels_first' == K.image_data_format():
            X = np.ones([self.batch_size, 1, self.img_w, self.img_h])
        else:
            X = np.ones([self.batch_size, self.img_w, self.img_h, 1])
        # batch-size lables have to be initialized
        # initialize each lable with `space`,
        # e.g. fill the vector for each label with the index of `space` in the alphabet which is `0`
        y = np.zeros([self.batch_size, self.max_text_len])
        input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
        label_length = np.zeros((self.batch_size, 1))
        # create batch
        for i, idx in enumerate(indexes):
            img = self.imgs[idx]
            text = self.labels[idx]
            img = img.T
            if 'channels_first' == K.image_data_format():
                img = np.expand_dims(img, 0)
            else:
                img = np.expand_dims(img, -1)
            X[i] = img
            y[i, 0:len(text)] = self._text_to_labels(text)
            label_length[i] = len(text)
        inputs = {
            'the_input': X,
            'the_labels': y,
            'input_length': input_length,
            'label_length': label_length,
        }
        outputs = {'ctc': np.zeros([self.batch_size])}
        return inputs, outputs

    def _is_valid_str(self, text):
        for c in text:
            if c not in self.alphabet:
                return False
        return True

    def _text_to_labels(self, text):
        return list(map(lambda x: self.alphabet.index(x), text))

    @property
    def output_size(self):
        # size of the alphabet + special `blank` of CTC
        return len(self.alphabet) + 1

    def on_epoch_end(self):
        # updates indexes after each epoch
        self.indexes = np.arange(self.n)
        np.random.shuffle(self.indexes)
