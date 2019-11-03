import absl.logging
import logging
import os
import sys
import tensorflow as tf
import time

from dotenv import load_dotenv, find_dotenv
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from pathlib import Path

from generator import DataGenerator
from model import OCRNet


os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["SKMP_SETTINGS"] = "TRUE"


def main():
    # env
    env_path = find_dotenv()
    load_dotenv(dotenv_path=env_path, verbose=True)
    processed_p = Path(os.environ.get('PATH_PROCESSED')).resolve()
    models_p = Path(os.environ.get('PATH_MODELS')).resolve()
    img_h = int(os.environ.get('IMAGE_HEIGHT'))
    img_w = int(os.environ.get('IMAGE_WIDTH'))
    batch_size = int(os.environ.get('BATCH_SIZE'))
    epochs = int(os.environ.get('EPOCHS'))
    ngpus = int(os.environ.get('NGPUS'))
    downsample_factor = int(os.environ.get('DOWNSAMPLE_FACTOR'))
    lr = float(os.environ.get('LEARNING_RATE'))
    # calculate batchsize according to strategy
    strategy = tf.distribute.MirroredStrategy() if 1 < ngpus else tf.distribute.OneDeviceStrategy(device="/gpu:1")
    batch_size = batch_size * strategy.num_replicas_in_sync
    # logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('TensorFlow version: ' + tf.__version__)
    logger.info('Keras version: ' + tf.keras.__version__)
    # parameters
    val_p = processed_p.joinpath('val')
    assert val_p.exists()
    train_p = processed_p.joinpath('train')
    assert train_p.exists()
    # generators
    logger.info('load data')
    train_gen = DataGenerator(train_p, img_w, img_h, batch_size, downsample_factor)
    val_gen = DataGenerator(val_p, img_w, img_h, batch_size, downsample_factor)
    assert train_gen.alphabet == val_gen.alphabet
    assert train_gen.output_size == val_gen.output_size
    max_text_len = max(train_gen.max_text_len, val_gen.max_text_len)
    logger.info('alphabet: \'' + str(train_gen.alphabet) + '\'')
    logger.info('alphabet size: ' + str(len(train_gen.alphabet)))
    logger.info('max text length: ' + str(max_text_len))
    logger.info('image shape: height=' + str(img_h) + ' width=' + str(img_w))
    logger.info('batch size: ' + str(batch_size))
    logger.info('output size: ' + str(train_gen.output_size))
    logger.info('training samples: ' + str(train_gen.n))
    logger.info('validation samples: ' + str(val_gen.n))
    logger.info('epochs: ' + str(epochs))
    logger.info('train steps per epoch: ' + str(len(train_gen)))
    logger.info('validation steps per epoch: ' + str(len(val_gen)))
    logger.info('learning rate: ' + str(lr))
    with strategy.scope():
        # create model
        model = OCRNet(train_gen.output_size, img_w, img_h, max_text_len)
        model.summary()
        # initialize optimizer
        adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # compile model
        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam, metrics=['accuracy'])
    # callbacks
    log_p = models_p.joinpath('logs')
    log_p.mkdir(exist_ok=True)
    callbacks = [
            EarlyStopping(patience=2, monitor='val_loss'),
            ModelCheckpoint(str(models_p.joinpath('model.h5')))
    ]
    # model training
    logger.info('start fitting model')
    start = time.perf_counter()
    model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            shuffle=False,
            use_multiprocessing=True,
            workers=6,
            callbacks=callbacks)
    elapsed = time.perf_counter() - start
    logger.info('elapsed: {:0.3f}'.format(elapsed))


if __name__ == '__main__':
    main()
