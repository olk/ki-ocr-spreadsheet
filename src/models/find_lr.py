import absl.logging
import logging
import os
import tensorflow as tf

from lrf import LRFinder
from dotenv import load_dotenv, find_dotenv
from tensorflow.keras.optimizers import Adam
from pathlib import Path

from generator import DataGenerator
from model import OCRNet


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
    min_lr = float(os.environ.get('MIN_LEARNING_RATE'))
    max_lr = float(os.environ.get('MAX_LEARNING_RATE'))
    # according how Keras' multi_gpu_mode() handles mini-batches
    # logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('TensorFlow version: ' + tf.__version__)
    logger.info('Keras version: ' + tf.keras.__version__)
    # parameters
    train_p = processed_p.joinpath('train')
    assert train_p.exists()
    # generators
    logger.info('loading data')
    train_gen = DataGenerator(train_p, img_w, img_h, batch_size, downsample_factor)
    max_text_len = train_gen.max_text_len
    logger.info('alphabet: \'' + str(train_gen.alphabet) + '\'')
    logger.info('alphabet size: ' + str(len(train_gen.alphabet)))
    logger.info('max text length: ' + str(max_text_len))
    logger.info('image shape: height=' + str(img_h) + ' width=' + str(img_w))
    logger.info('batch size: ' + str(batch_size))
    logger.info('output size: ' + str(train_gen.output_size))
    logger.info('training samples: ' + str(train_gen.n))
    logger.info('train steps per epoch: ' + str(len(train_gen)))
    logger.info('min. learning-rate: ' + str(min_lr))
    logger.info('max. learning-rate: ' + str(max_lr))
    # create model
    model = OCRNet(train_gen.output_size, img_w, img_h, max_text_len)
    model.summary()
    # find best learning rate
    # initialize optimizer
    adam = Adam(lr=min_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # compile model
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])
    lrf = LRFinder(model)
    lrf.find(train_gen,
             min_lr, max_lr,
             stepsPerEpoch=len(train_gen),
             batchSize=batch_size)
    # plot the loss for the various learning rates and save the
    # resulting plot to disk
    if not models_p.exists():
        models_p.mkdir()
    lrf.plot_loss(models_p.joinpath('loss_plot.png'), title='loss')
    lrf.plot_loss_change(models_p.joinpath('loss_change_plot.png'), title='loss change')
    # in the config and then train the network for our full set of
    logger.info('learning rate finder complete')
    logger.info('best LR: %f' % lrf.get_best_lr())


if __name__ == '__main__':
    main()
