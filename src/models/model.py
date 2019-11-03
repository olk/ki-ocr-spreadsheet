from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import add, concatenate
from tensorflow.keras.layers import Activation, Conv2D, Dense, GRU, Input, Lambda, Layer, MaxPooling2D, Reshape


class FeatureExtraction(Layer):
    def __init__(self, conv_filters, pool_size, name='feature-extraction', **kwargs):
        super(FeatureExtraction, self).__init__(name=name, **kwargs)
        self.conv1 = Conv2D(filters=conv_filters, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')
        self.conv2 = Conv2D(filters=conv_filters, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')
        self.max1 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')
        self.max2 = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        return self.max2(x)

    def get_config(self):
        return super(FeatureExtraction, self).get_config()


class FeatureReduction(Layer):
    def __init__(self, img_w, img_h, pool_size, conv_filters, name='feature-reduction', **kwargs):
        super(FeatureReduction, self).__init__(name=name, **kwargs)
        target_shape = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
        self.reshape = Reshape(target_shape=target_shape, name='reshape')
        self.dense = Dense(32, activation='relu', name='dense')

    def call(self, inputs):
        x = self.reshape(inputs)
        return self.dense(x)

    def get_config(self):
        return super(FeatureReduction, self).get_config()


class SequentialLearner(Layer):
    def __init__(self, name='sequential-learner', **kwargs):
        super(SequentialLearner, self).__init__(name=name, **kwargs)
        self.gru_1a = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru_1a')
        self.gru_1b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_1b')
        self.gru_2a = GRU(512, return_sequences=True, kernel_initializer='he_normal', name='gru_2a')
        self.gru_2b = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru_2b')

    def call(self, inputs):
        x_1a = self.gru_1a(inputs)
        x_1b = self.gru_1b(inputs)
        x = add([x_1a, x_1b])
        x_2a = self.gru_2a(x)
        x_2b = self.gru_2b(x)
        return concatenate([x_2a, x_2b])

    def get_config(self):
        return super(SequentialLearner, self).get_config()


class Output(Layer):
    def __init__(self, output_size, name='output', **kwargs):
        super(Output, self).__init__(name=name, **kwargs)
        self.dense = Dense(output_size, kernel_initializer='he_normal', name='dense')
        self.softmax = Activation('softmax', name='softmax')

    def call(self, inputs):
        x = self.dense(inputs)
        return self.softmax(x)

    def get_config(self):
        return super(Output, self).get_config()


class OCRNet(Model):
    def __init__(self, output_size, img_w, img_h, max_text_len, name='OCRNet', **kwargs):
        # parameters
        conv_filters = 16
        pool_size = 2
        # define layers
        feature_extraction = FeatureExtraction(conv_filters=conv_filters, pool_size=pool_size)
        sequential_learner = SequentialLearner()
        feature_reduction = FeatureReduction(img_w=img_w, img_h=img_h, pool_size=pool_size, conv_filters=conv_filters)
        output = Output(output_size)
        # NHWC == channels_last NCHW == channels_first
        # initialize input shape
        if 'channels_first' == K.image_data_format():
            input_shape = (1, img_w, img_h)
        else:
            input_shape = (img_w, img_h, 1)
        # input
        inputs = Input(name='the_input', shape=input_shape, dtype='float32')
        labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # call layers
        x = feature_extraction(inputs)
        x = feature_reduction(x)
        x = sequential_learner(x)
        predictions = output(x)
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self._ctc_lambda_func, output_shape=(1,), name='ctc')([predictions, labels, input_length, label_length])
        super(OCRNet, self).__init__(
                inputs=[inputs, labels, input_length, label_length], outputs=loss_out,
                name=name, **kwargs)

        # ctc decoder
        flattened_input_length = K.reshape(input_length, (-1,))
        top_k_decoded, _ = K.ctc_decode(predictions, flattened_input_length)
        self.decoder = K.function([inputs, flattened_input_length], [top_k_decoded[0]])

    # loss and train functions, network architecture
    def _ctc_lambda_func(self, args):
        predictions, labels, input_length, label_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage
        predictions = predictions[:, 2:, :]
        return K.ctc_batch_cost(labels, predictions, input_length, label_length)
