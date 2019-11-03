import absl.logging
import itertools
import logging
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from dotenv import load_dotenv, find_dotenv
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

from generator import DataGenerator
from model import OCRNet


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


def main():
    # env
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    models_p = Path(os.environ.get('PATH_MODELS')).resolve()
    img_h = int(os.environ.get('IMAGE_HEIGHT'))
    img_w = int(os.environ.get('IMAGE_WIDTH'))
    batch_size = int(os.environ.get('BATCH_SIZE'))
    downsample_factor = int(os.environ.get('DOWNSAMPLE_FACTOR'))
    lr = float(os.environ.get('LEARNING_RATE'))
    # logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('TensorFlow version: ' + tf.__version__)
    logger.info('Keras version: ' + tf.keras.__version__)
    # parameters
    test_p = processed_p.joinpath('test')
    assert test_p.exists()
    logger.info('load data')
    test_gen = DataGenerator(test_p, img_w, img_h, batch_size, downsample_factor)
    alphabet = test_gen.alphabet
    logger.info('image shape: height=' + str(img_h) + ' width=' + str(img_w))
    logger.info('batch size: ' + str(batch_size))
    logger.info('test samples: ' + str(test_gen.n))
    logger.info('test steps per epoch: ' + str(len(test_gen)))
    logger.info('learning rate: ' + str(lr))
    # model
    checkpoint_p = models_p.joinpath('model.h5')
    assert checkpoint_p.exists()
    model = load_model(str(checkpoint_p), compile=False)
    model.summary()
    logger.info('model loaded')
    # optimizer
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])
    logger.info('model compiled')
    # test data
    score = model.evaluate_generator(
            generator=test_gen,
            steps=len(test_gen),
            verbose=1)
    logger.info('loss %.3f accuracy: %.3f' % (score[0], score[1]))


if __name__ == '__main__':
    main()
