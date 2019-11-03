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

import asyncio
import csv
import cv2
import imutils
import logging
import numpy as np
import os
import shutil

from dotenv import load_dotenv, find_dotenv
from pathlib import Path


def split_data(data_size, jpgs, train_frac, val_frac):
    assert data_size == len(jpgs)
    # create array of indices
    # each index represents one spreadsheet (e.g. jpg)
    indices = np.arange(0, data_size)
    # split indices in training/validation/testing subsets
    train_indices, test_indices, val_indices = np.split(indices, [int(train_frac * len(indices)), int((1 - val_frac) * len(indices))])
    # split jpgs according to the indices
    train_jpgs = jpgs[train_indices[0]:train_indices[-1]+1]
    test_jpgs = jpgs[test_indices[0]:test_indices[-1]+1]
    val_jpgs = jpgs[val_indices[0]:val_indices[-1]+1]
    return train_jpgs, val_jpgs, test_jpgs


async def async_exec(cmd):
    create = asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE)
    proc = await create
    try:
        await proc.wait()
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


def split_jpg(loop, chunk, dir_p):
    tasks = []
    for jpg_p in chunk:
        tasks.append(
                loop.create_task(
                    async_exec('src/features/split --file ' + str(jpg_p) + ' --outdir ' + str(dir_p))))
    loop.run_until_complete(asyncio.wait(tasks))


def create_label(csv_p, writer):
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
            skipped_cols = 0
            for j, label in enumerate(row):
                # skipp `of` column
                if 12 == j:
                    skipped_cols += 1
                    continue
                jpg = ('%s-%d-%d.jpg' % (str(csv_p.stem), i - skipped_rows + 1, j - skipped_cols + 1))
                writer.writerow({'image': jpg, 'label': label})


def pad_and_resize(dir_p, img_h, img_w):
    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    # loop over the input image paths
    for img_p in dir_p.glob('*.jpg'):
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
                pad_h = max(32 - h, 0)
                pad_w = max(128 - w, 0)
                if pad_h > 0 or pad_w > 0:
                    roi = cv2.copyMakeBorder(roi, pad_h // 2, pad_h // 2, pad_w // 2,
                                             pad_w // 2, cv2.BORDER_CONSTANT,
                                             value=(255, 255, 255))
                roi = cv2.resize(roi, (img_w, img_h))
                cv2.imwrite(str(img_p), roi)


def process(loop, jpgs_p, dir_p, chunk_size):
    # create images
    # executed async. + parallel prevents exhausting system resources
    for chunk in [jpgs_p[i:i+chunk_size] for i in range(0, len(jpgs_p), chunk_size)]:
        split_jpg(loop, chunk, dir_p)
    # create labels
    labels_p = dir_p.joinpath('labels.csv')
    with open(labels_p, 'w+') as f:
        writer = csv.DictWriter(f, fieldnames=('image', 'label'))
        writer.writeheader()
        for jpg_p in jpgs_p:
            create_label(jpg_p.with_suffix('.csv'), writer)


def main():
    # environment
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    # parameters
    raw_p = Path(os.environ.get('PATH_RAW')).resolve()
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    data_size = int(os.environ.get('DATA_SIZE'))
    chunk_size = int(os.environ.get('CHUNK_SIZE'))
    img_h = int(os.environ.get('IMAGE_HEIGHT'))
    img_w = int(os.environ.get('IMAGE_WIDTH'))
    train_frac = float(os.environ.get('TRAIN_FRAC'))
    val_frac = float(os.environ.get('VAL_FRAC'))
    # logging
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    # prepare directories
    shutil.rmtree(processed_p, ignore_errors=True)
    processed_p.mkdir()
    # process PDFs
    jpgs_p = sorted(list(raw_p.glob('*.jpg')), key=lambda f: int(f.stem))
    assert len(jpgs_p) == len(list(raw_p.glob('*.csv')))
    # create training data
    loop = asyncio.get_event_loop()
    # split data into training, validation and test
    train_jpgs_p, val_jpgs_p, test_jpgs_p = split_data(data_size, jpgs_p, train_frac, val_frac)
    # create training data
    train_p = processed_p.joinpath('train')
    train_p.mkdir()
    process(loop, train_jpgs_p, train_p, chunk_size)
    pad_and_resize(train_p, img_h, img_w)
    logger.info('training data generated')
    # create validation data
    val_p = processed_p.joinpath('val')
    val_p.mkdir()
    process(loop, val_jpgs_p, val_p, chunk_size)
    pad_and_resize(val_p, img_h, img_w)
    logger.info('validation data generated')
    # create test data
    test_p = processed_p.joinpath('test')
    test_p.mkdir()
    process(loop, test_jpgs_p, test_p, chunk_size)
    pad_and_resize(test_p, img_h, img_w)
    logger.info('test data generated')
    loop.close()


if __name__ == '__main__':
    main()
