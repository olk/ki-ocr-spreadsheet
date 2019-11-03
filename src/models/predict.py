import absl.logging
import argparse
import csv
import cv2
import imutils
import itertools
import logging
import numpy as np
import os
import subprocess
import tensorflow as tf

from dotenv import load_dotenv, find_dotenv
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

from generator import DataGenerator
from parallel import ParallelModel

def trim(jpg_p):
    cmd = "src/features/trim --file " + str(jpg_p)
    subprocess.check_call(cmd, shell=True)

def split(jpg_p, outdir_p):
    cmd = "src/features/split --file " + str(jpg_p) + " --outdir " + str(outdir_p)
    subprocess.check_call(cmd, shell=True)

def pad_and_resize(dir_p, img_h, img_w):
    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 23))
    # loop over the input image paths
    for img_p in dir_p.glob('*.jpg'):
        # load the image, resize it, and convert it to grayscale
        img = cv2.imread(str(img_p))
        #img = imutils.resize(img, height=64, width=512)
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
        thresh = cv2.threshold(gradX, 0, 255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
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
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
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
                                pad_w // 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    roi = cv2.resize(roi, (img_w, img_h))
                    cv2.imwrite(str(img_p), roi)


def build_model(ncores, lr, models_p):
    # model
    checkpoint_p = models_p.joinpath('model.h5')
    assert checkpoint_p.exists()
    model = load_model(str(checkpoint_p), compile=False)
    model.summary()
    # optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])
    input_data = model.get_layer('the_input').output
    y_pred = model.get_layer('softmax').output
    model = Model(inputs=input_data, outputs=y_pred)
    return model


# function to decode neural network output
# for a real OCR application, this should be
# beam search with a dictionary and language model
def decode_batch(out, alphabet):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(alphabet):
                outstr += alphabet[c]
        ret.append(outstr)
    return ret


def main(jpg_p, outdir_p):
    # env
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    models_p = Path(os.environ.get('PATH_MODELS')).resolve()
    img_h = int(os.environ.get('IMAGE_HEIGHT'))
    img_w = int(os.environ.get('IMAGE_WIDTH'))
    batch_size = int(os.environ.get('BATCH_SIZE'))
    downsample_factor = int(os.environ.get('DOWNSAMPLE_FACTOR'))
    ncores = int(os.environ.get('NCORES'))
    lr = float(os.environ.get('LEARNING_RATE'))
    # logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('TensorFlow version: ' + tf.__version__)
    logger.info('Keras version: ' + tf.keras.__version__)
    # preparation
    train_p = processed_p.joinpath('train')
    assert train_p.exists()
    assert jpg_p.exists()
    if not outdir_p.exists():
        outdir_p.mkdir()
    split(jpg_p, outdir_p)
    pad_and_resize(outdir_pi, img_h, img_w)
    # model
    model = build_model(ncores, lr, models_p)
    # process
    train_gen = DataGenerator(train_p, img_w, img_h, batch_size, downsample_factor)
    stem = jpg_p.stem
    with open(outdir_p.joinpath(stem + '.csv'), 'w+') as csv_f:
        writer = csv.writer(csv_f, delimiter=';')
        # CSV header
        writer.writerow((
                'Symbol',
                'High',
                'Low',
                'Now',
                'Sell Target Potential',
                'Worst-Case Drawdowns',
                'Range Index',
                'Win Odds/100',
                '% Payoff',
                'Days Held',
                'Annual Rate of Return',
                'Sample Size',
                '',
                'Creadible Ratio',
                'Rwd~Rsk Ratio',
                'Wghted'))
        for r in range(1, 21):
            row = []
            for c in range(1, 17):
                if 'channels_first' == K.image_data_format():
                    X = np.ones([1, 1, img_w, img_h])
                else:
                    X = np.ones([1, img_w, img_h, 1])
                img_p = outdir_p.joinpath('%s-%d-%d.jpg' % (stem, r, c))
                assert img_p.exists()
                img = cv2.imread(str(img_p))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32)
                img /= 255
                img = img.T
                if 'channels_first' == K.image_data_format():
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X[0] = img
                y = np.zeros([1, train_gen.max_text_len])
                input_length = np.ones((1, 1)) * (img_w // downsample_factor - 2)
                label_length = np.zeros((1, 1))
                inputs = {
                    'the_input': X,
                    'the_labels': y,
                    'input_length': input_length,
                    'label_length': label_length,
                    'the_sources': img_p.name,
                }
                net_out_value = model.predict(inputs)
                pred_texts = decode_batch(net_out_value, train_gen.alphabet)
                row.append(pred_texts[0])
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-o', '--outdir', type=str, required=True)
    args = parser.parse_args()
    main(Path(args.file), Path(args.outdir))
