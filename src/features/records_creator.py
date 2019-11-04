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

import csv
import cv2
import imutils
import logging
import numpy as np
import os
import shutil
import subprocess
import tempfile
import tensorflow as tf

from pathlib import Path
from tensorflow.compat import as_bytes


class OCRTFRecordsCreator(object):
    def __init__(self, cohort_size, alphabet, max_text_len, img_h=32, img_w=128):
        self._cohort_size = cohort_size
        self._alphabet = alphabet
        self._max_text_len = max_text_len
        self._img_h = img_h
        self._img_w = img_w

    def _bytes_feature(self, value):
        if not isinstance(value, list):
          value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(self, value):
        if not isinstance(value, list):
          value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _exec(self, cmd):
        subprocess.check_call(cmd, shell=True)


    def _pad_image(self, img_p):
        # initialize a rectangular and square structuring kernel
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
        # loop over the input image paths
        # load the image, resize it, and convert it to grayscale
        img = cv2.imread(str(img_p))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # smooth the image using a 3x3 Gaussian, then apply the blackhat
        # morphological operator to find dark regions on a light background
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        # compute the Scharr gradient of the blackhat image and scale the
        # result into the range [0, 255]
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        # apply a closing operation using the rectangular kernel to close
        # gaps in between letters -- then apply Otsu's thresholding method
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # perform another closing operation, this time using the square
        # kernel to close gaps between lines of the MRZ, then perform a
        # serieso of erosions to break apart connected components
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.erode(thresh, None, iterations=4)
        # during thresholding, it's possible that border pixels were
        # included in the thresholding, so let's set 5% of the left and
        # right borders to zero
        p = int(img.shape[1] * 0.05)
        thresh[:, 0:p] = 0
        thresh[:, img.shape[1] - p:] = 0
        # find contours in the thresholded image and sort them by their
        # size
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # loop over the contours
        assert 1 <= len(cnts)
        for c in cnts:
            # compute the bounding box of the contour and use the contour to
            # compute the aspect ratio and coverage ratio of the bounding box
            # width to the width of the image
            (x, y, w, h) = cv2.boundingRect(c)
            # check to see if the aspect ratio and coverage width are within
            # acceptable criteria
            if 1 <= w and 2 <= h:
                # pad the bounding box since we applied erosions and now need
                # to re-grow it
                pX = int((x + w) * 0.03)
                pY = int((y + h) * 0.03)
                (x, y) = (x - 3 * pX, y - 8 * pY)
                (w, h) = (w + 7 * pX, h + 6 * pY)
                # extract the ROI from the image and draw a bounding box
                # surrounding the MRZ
                roi = img[y:y + h, x:x + w].copy()
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                h, w, _ = roi.shape
                pad_h = max(self._img_h - h, 0)
                pad_w = max(self._img_w - w, 0)
                if pad_h > 0 or pad_w > 0:
                    roi = cv2.copyMakeBorder(roi, pad_h // 2, pad_h // 2, pad_w // 2,
                                             pad_w // 2, cv2.BORDER_CONSTANT,
                                             value=(255, 255, 255))
                return cv2.resize(roi, (self._img_w, self._img_h))

    def _convert_to_tfrecord(self, cohort, dir_p, record_writer):
        images_features = []
        labels_features = []
        for jpg_p in cohort:
            assert jpg_p.exists()
            csv_p = jpg_p.with_suffix('.csv')
            assert csv_p.exists()
            stem = int(jpg_p.stem)
            with tempfile.TemporaryDirectory() as outdir:
                # split spreadsheet in cells
                self._exec('src/features/split --file ' + str(jpg_p) + ' --outdir ' + outdir)
                # read lables
                with open(csv_p) as csv_f:
                    reader = csv.reader(csv_f, delimiter=';')
                    skipped_rows = 0
                    for i, row in enumerate(reader):
                        # skip header
                        if 3 > i or 13 == i:
                            skipped_rows += 1
                            continue
                        # skip summary
                        if 23 < i:
                            break
                        r = i - skipped_rows + 1
                        skipped_cols = 0
                        for j, label in enumerate(row):
                            # skipp `of` column
                            if 12 == j:
                                skipped_cols += 1
                                continue
                            c = j - skipped_cols + 1
                            cell_p = Path('%s/%d-%d-%d.jpg' % (outdir, stem, r, c))
                            assert cell_p.exists()
                            # pad images
                            img = self._pad_image(cell_p)
                            assert img is not None
                            # sanity checks
                            height, width, _ = img.shape
                            assert height == self._img_h
                            assert width == self._img_w

                            # decode JPEG to RGB grid of pixels (gray)
                         #  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                         #  # convert to floating-point tensor
                         #  img = img.astype(np.float32)
                         #  # rescale pixel values to the [0,1] interval
                         #  img /= 255
                         #  img = img.T

                            # convert image to list of bytes
                            img_raw = img.tostring()

                            images_features.append(self._bytes_feature(img_raw))
                            labels_features.append(self._bytes_feature(label.encode('utf-8')))

        ctx = tf.train.Features(
                feature={
                    'alphabet': self._bytes_feature(self._alphabet.encode('utf-8')),
                    'max_text_len': self._int64_feature(self._max_text_len),
                    'img_h': self._int64_feature(self._img_h),
                    'img_w': self._int64_feature(self._img_w)
                })
        data = tf.train.FeatureLists(
                feature_list={
                    'images': tf.train.FeatureList(feature=images_features),
                    'labels': tf.train.FeatureList(feature=labels_features)
                })

        ex= tf.train.SequenceExample(context=ctx, feature_lists=data)
        # write record
        record_writer.write(ex.SerializeToString())

    def _process_cohort(self, dir_p, cohort, idx):
        tfr_p = dir_p.joinpath('ocr%d.tfr' % idx)
        print('creating %s' % str(tfr_p))
        with tf.io.TFRecordWriter(str(tfr_p)) as record_writer:
            self._convert_to_tfrecord(cohort, dir_p, record_writer)

    def create(self, dir_p, jpgs_p):
        assert 0 < len(jpgs_p)
        if not dir_p.exists():
            dir_p.mkdir()
        for idx, cohort in enumerate([jpgs_p[i:i+self._cohort_size] for i in range(0, len(jpgs_p), self._cohort_size)]):
            self._process_cohort(dir_p, cohort, idx + 1)
