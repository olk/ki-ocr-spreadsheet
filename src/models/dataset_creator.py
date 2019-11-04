'''
                    Copyright Oliver Kowalke 2018.
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

from pathlib import Path


class OCRTFDatasetCreator(object):
    def __init__(self, dir_p, batch_size, img_h=32, img_w=128, train=True):
        self.n = 0
        self._img_h = img_h
        self._img_w = img_w
        tfrecords_files = [str(f) for f in dir_p.glob('*.tfr')]
        # read dataset from tfrecords
        ds = tf.data.TFRecordDataset(tfrecords_files)
        # decode/parse 
        dataset = ds.map(self._parse_and_decode, num_parallel_calls=batch_size)
        if train:
            # For training dataset, do a shuffle and repeat
            self.dataset = dataset.shuffle(1000).repeat().batch(batch_size)        
        else:
            self.dataset = dataset.batch(batch_size)

    def _parse_and_decode(self, serialized_ex):
        ex = tf.io.parse_single_example(
            serialized_ex,
            features={
              'image': tf.io.FixedLenFeature([], tf.string),
              'label': tf.io.FixedLenFeature([], tf.string),
              'height': tf.io.FixedLenFeature([], tf.int64),
              'width': tf.io.FixedLenFeature([], tf.int64),
              'depth': tf.io.FixedLenFeature([], tf.int64),
              }
            )
        img = tf.io.parse_tensor(ex['image'], tf.float32)
        # reshape to channels, height, width and
        img_shape = [ex['height'], ex['width'], ex['depth']]
        img = tf.reshape(img, img_shape)
        label = tf.cast(ex['label'], tf.string)
        #label = tf.one_hot(label, 2)

        return img, label
