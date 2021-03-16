import tensorflow as tf
import CustomMetrics

from kerastuner import HyperModel

class Arquitectura1(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.SimpleRNN(
                input_shape=self.input_shape,
                units=hp.Int(
                    "simple_rnn_units", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                return_sequences=False
            )
        )
        
        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura2(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_rnn_units", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                return_sequences=False
            )
        )
        
        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura3(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                return_sequences=False
            )
        )
        
        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura7(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.Conv1D(
                input_shape=self.input_shape,
                filters=hp.Int(
                    "filters_conv1", min_value=8, max_value=64, step=8, default=32
                ),
                kernel_size=hp.Int(
                    "kernel_conv1", min_value=1, max_value=4, step=1, default=1
                ),
                strides=hp.Int(
                    "stride_conv1", min_value=1, max_value=2, step=1, default=1
                )
            )
        )

        model.add(tf.keras.layers.BatchNormalization())
        model.add(
            tf.keras.layers.Activation(
                activation=hp.Choice(
                    "activation1",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )
        
        model.add(
            tf.keras.layers.MaxPool1D(
                pool_size=hp.Int(
                    "max_pool1", min_value=1, max_value=4, step=1, default=1
                )
            )
        )

        model.add(tf.keras.layers.GlobalAveragePooling1D())

        model.add(
            tf.keras.layers.Dropout(
                rate=hp.Float(
                    'dropout', min_value=0.0, max_value=0.5, step=0.05, default=0.25
                )
            )
        )

        model.add(
            tf.keras.layers.Dense(
                units=self.output_units,
                activation=hp.Choice(
                    "output_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu",
                )
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class Arquitectura8(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "lstm_layer_2_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura9(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "gru_layer_2_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class Arquitectura10(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "lstm_layer_1_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class Arquitectura11(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "gru_layer_1_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura12(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "lstm_layer_2_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )
        
        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura13(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "gru_layer_2_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura14(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "lstm_layer_1_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )
        
        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura15(HyperModel):
    def __init__(self,input_shape,output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                recurrent_dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=hp.Choice(
                    "gru_layer_1_sequence",
                    values=[True,False],
                    default=True
                )
            )
        )

        model.add(tf.keras.layers.Dense(units=self.output_units,activation=None))

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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura16(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura17(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )
        
        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class Arquitectura18(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model


class Arquitectura19(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura20(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )
        
        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura21(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura22(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura23(HyperModel):
    def __init__(self,input_shape,output_units,n_steps_out):
        self.input_shape = input_shape
        self.output_units = output_units
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(tf.keras.layers.RepeatVector(self.n_steps_out))

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=self.output_units,activation=None)
            )
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
            loss=tf.losses.MeanSquaredError(),
            metrics=[
                tf.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanAbsolutePercentageError(),
                CustomMetrics.symmetric_mean_absolute_percentage_error],
        )

        return model

class Arquitectura31(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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

class Arquitectura32(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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


class Arquitectura33(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )
        
        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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


class Arquitectura34(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        "dense_layer", min_value=24, max_value=120, step=24, default=48
                    ),
                    activation=hp.Choice(
                        "dense_layer_activation",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"
                    )
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
                )
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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

class Arquitectura35(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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

class Arquitectura36(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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

class Arquitectura37(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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

class Arquitectura38(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.Flatten()
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

class Arquitectura39(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
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

class Arquitectura40(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
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

class Arquitectura41(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )
        
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
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


class Arquitectura42(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
            )
        )

        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(
                    "dense_layer", min_value=24, max_value=120, step=24, default=48
                ),
                activation=hp.Choice(
                    "dense_layer_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"
                )
            )
        )

        model.add(tf.keras.layers.Dropout(
            rate=hp.Float(
                "dropout_dense",
                min_value=0,
                max_value=0.99,
                step=0.09,
                default=0
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

class Arquitectura43(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.LSTM(   
                units=hp.Int(
                    "lstm_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
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

class Arquitectura44(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=True
            )
        )

        model.add(
            tf.keras.layers.GRU(   
                units=hp.Int(
                    "gru_units_layer_2", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_2",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_2",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
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

class Arquitectura45(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.LSTM(
                input_shape=self.input_shape,   
                units=hp.Int(
                    "lstm_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
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

class Arquitectura46(HyperModel):
    def __init__(self,input_shape,n_steps_out):
        self.input_shape = input_shape
        self.n_steps_out = n_steps_out

    def build(self, hp):

        model = tf.keras.Sequential()

        model.add(
            tf.keras.layers.GRU(
                input_shape=self.input_shape,
                units=hp.Int(
                    "gru_units_layer_1", min_value=64, max_value=512, step=64, default=128
                ),
                activation='tanh',
                kernel_regularizer=tf.keras.regularizers.L1(
                    l1=hp.Float(
                        "kernel_regularizer_layer_1",
                        min_value=0,
                        max_value=0.105,
                        step=0.0075,
                        default=1e-2,
                        )
                    ),
                dropout=hp.Float(
                    "dropout_regularizer_layer_1",
                    min_value=0,
                    max_value=0.99,
                    step=0.09,
                    default=0
                ),
                return_sequences=False
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