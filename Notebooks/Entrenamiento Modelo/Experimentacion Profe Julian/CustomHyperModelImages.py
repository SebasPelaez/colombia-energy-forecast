import tensorflow as tf
import CustomMetrics

from kerastuner import HyperModel

class ArquitecturaI1(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.ConvLSTM2D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "convLSTM2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_1", min_value=3, max_value=5, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_1",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.ConvLSTM2D(
                filters=hp.Int(
                    "convLSTM2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_3", min_value=3, max_value=5, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_3",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.ConvLSTM2D(
                filters=hp.Int(
                    "convLSTM2d_filters_layer_5", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_5", min_value=3, max_value=5, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_5",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=hp.Int(
                    "pool2d_size_layer_6", min_value=3, max_value=5, step=2, default=3
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_units_layer_8", min_value=24, max_value=120, step=24, default=120
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.n_steps_out,activation=None))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI2(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.ConvLSTM2D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "convLSTM2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_1",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool2d_size_layer_2", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.ConvLSTM2D(
                filters=hp.Int(
                    "convLSTM2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_3", min_value=3, max_value=7, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_3",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=hp.Int(
                    "pool2d_size_layer_4", min_value=3, max_value=5, step=2, default=3
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_units_layer_6", min_value=24, max_value=120, step=24, default=120
                ),
                activation=hp.Choice(
                    "dense_layer_6_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.n_steps_out,activation=None))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI3(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.ConvLSTM2D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "convLSTM2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_1",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=hp.Int(
                    "pool2d_size_layer_2", min_value=3, max_value=7, step=2, default=3
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_units_layer_4", min_value=24, max_value=120, step=24, default=120
                ),
                activation=hp.Choice(
                    "dense_layer_4_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.n_steps_out,activation=None))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI4(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.ConvLSTM2D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "convLSTM2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_1", min_value=3, max_value=5, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_1",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.ConvLSTM2D(
                filters=hp.Int(
                    "convLSTM2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_3", min_value=3, max_value=5, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_3",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.ConvLSTM2D(
                filters=hp.Int(
                    "convLSTM2d_filters_layer_5", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_5", min_value=3, max_value=5, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_5",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=hp.Int(
                    "pool2d_size_layer_6", min_value=3, max_value=5, step=2, default=3
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
        )

        model.add(tf.keras.layers.Dense(units=self.n_steps_out,activation=None))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI5(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.ConvLSTM2D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "convLSTM2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_1",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool2d_size_layer_2", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.ConvLSTM2D(
                filters=hp.Int(
                    "convLSTM2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_3", min_value=3, max_value=7, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_3",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=hp.Int(
                    "pool2d_size_layer_4", min_value=3, max_value=5, step=2, default=3
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
        )

        model.add(tf.keras.layers.Dense(units=self.n_steps_out,activation=None))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI6(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.ConvLSTM2D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "convLSTM2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                ),
                kernel_size=hp.Int(
                    "convLSTM2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                ),
                activation='relu',
                padding=hp.Choice(
                    "conv2d_padding_layer_1",
                    values=["valid", "same"],
                    default="valid"
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.MaxPooling2D(
                pool_size=hp.Int(
                    "pool2d_size_layer_2", min_value=3, max_value=7, step=2, default=3
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
        )

        model.add(tf.keras.layers.Dense(units=self.n_steps_out,activation=None))

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI7(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    input_shape=self.input_shape,
                    filters=hp.Int(
                        "conv2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_1", min_value=3, max_value=5, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_1",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        "conv2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_3", min_value=3, max_value=5, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_3",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        "conv2d_filters_layer_5", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_5", min_value=3, max_value=5, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_5",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_6", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )
        )

        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(
                    "lstm_units_layer_7", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_7",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                    )
                ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_7",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False,
                stateful=False
            )
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_units_layer_8", min_value=24, max_value=120, step=24, default=120
                ),
                activation=hp.Choice(
                    "dense_layer_8_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(
            tf.keras.layers.Dense(units=self.n_steps_out,activation=None)
        )

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI8(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    input_shape=self.input_shape,
                    filters=hp.Int(
                        "conv2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_1",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_2", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        "conv2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_3", min_value=3, max_value=7, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_3",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_4", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )
        )

        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(
                    "lstm_units_layer_6", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_6",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                    )
                ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_6",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False,
                stateful=False
            )
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_units_layer_7", min_value=24, max_value=120, step=24, default=120
                ),
                activation=hp.Choice(
                    "dense_layer_7_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(
            tf.keras.layers.Dense(units=self.n_steps_out,activation=None)
        )

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class ArquitecturaI9(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    input_shape=self.input_shape,
                    filters=hp.Int(
                        "conv2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_1",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_2", min_value=3, max_value=7, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )
        )

        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(
                    "lstm_units_layer_3", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_4",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                    )
                ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_4",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False,
                stateful=False
            )
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_units_layer_5", min_value=24, max_value=120, step=24, default=120
                ),
                activation=hp.Choice(
                    "dense_layer_5_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(
            tf.keras.layers.Dense(units=self.n_steps_out,activation=None)
        )

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI10(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    input_shape=self.input_shape,
                    filters=hp.Int(
                        "conv2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_1", min_value=3, max_value=5, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_1",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        "conv2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_3", min_value=3, max_value=5, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_3",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        "conv2d_filters_layer_5", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_5", min_value=3, max_value=5, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_5",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_6", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )
        )

        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(
                    "lstm_units_layer_7", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_7",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                    )
                ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_7",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False,
                stateful=False
            )
        )

        model.add(
            tf.keras.layers.Dense(units=self.n_steps_out,activation=None)
        )

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class ArquitecturaI11(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    input_shape=self.input_shape,
                    filters=hp.Int(
                        "conv2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_1",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_2", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    filters=hp.Int(
                        "conv2d_filters_layer_3", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_3", min_value=3, max_value=7, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_3",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_4", min_value=3, max_value=5, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )
        )

        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(
                    "lstm_units_layer_6", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_6",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                    )
                ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_6",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False,
                stateful=False
            )
        )

        model.add(
            tf.keras.layers.Dense(units=self.n_steps_out,activation=None)
        )

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class ArquitecturaI12(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(
                    input_shape=self.input_shape,
                    filters=hp.Int(
                        "conv2d_filters_layer_1", min_value=8, max_value=32, step=32, default=32
                    ),
                    kernel_size=hp.Int(
                        "conv2d_kernel_layer_1", min_value=3, max_value=7, step=2, default=3
                    ),
                    activation='relu',
                    padding=hp.Choice(
                        "conv2d_padding_layer_1",
                        values=["valid", "same"],
                        default="valid"
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.MaxPooling2D(
                    pool_size=hp.Int(
                        "pool_kernel_layer_2", min_value=3, max_value=7, step=2, default=3
                    )
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Flatten()
            )
        )

        model.add(
            tf.keras.layers.LSTM(
                units=hp.Int(
                    "lstm_units_layer_3", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_4",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                    )
                ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_4",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False,
                stateful=False
            )
        )

        model.add(
            tf.keras.layers.Dense(units=self.n_steps_out,activation=None)
        )

        model.compile(
            optimizer=tf.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model