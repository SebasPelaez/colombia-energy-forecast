import tensorflow as tf
import CustomMetrics

from kerastuner import HyperModel

from kerastuner import HyperModel
from kerastuner.tuners import BayesianOptimization

class ModeloCompletoI_Concat_Version5(HyperModel):
	"""
	Este modelo corresponde a la combinación de los mejores modelo de experimentación
	en su versión #5, tanto para Datos Horarios, Diarios e Imagenes.
	Steps_In_Diario: 3
	Steps_In_Horario: 72
	Steps_out: 24
	"""
	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):

		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):
		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0.99,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=120,activation='relu'))(model_1_1)
		model_1_1 = tf.keras.layers.Dropout(rate=0)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(
		        units=512,
		        activation='tanh',
		        kernel_regularizer=tf.keras.regularizers.L1(l1=0),
		        dropout=0.54,
		        return_sequences=False
		)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(
		    tf.keras.layers.Conv2D(
		        filters=44,
		        kernel_size=3,
		        activation='relu',
		        padding='valid'
		    )
		)(input_3)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(
		    tf.keras.layers.Conv2D(
		        filters=20,
		        kernel_size=7,
		        activation='relu',
		        padding='valid'
		    )
		)(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(
		        units=64,
		        activation='tanh',
		        kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),
		        dropout=0.09,
		        return_sequences=False,
		        stateful=False
		)(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation='relu')(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation='relu')(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
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

		return full_model

class ModeloCompletoI_Suma_Version5(HyperModel):
	"""
	Este modelo corresponde a la combinación de los mejores modelo de experimentación
	en su versión #5, tanto para Datos Horarios, Diarios e Imagenes.
	Steps_In_Diario: 3
	Steps_In_Horario: 72
	Steps_out: 24
	"""
	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0.99,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=120,activation='relu'))(model_1_1)
		model_1_1 = tf.keras.layers.Dropout(rate=0)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.54,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=44,kernel_size=3,activation='relu',padding='valid'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=20,kernel_size=7,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.AveragePooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)

		model_3 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.09,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])
		full_model.compile(optimizer=tf.optimizers.Adam(hp.Float("learning_rate",min_value=1e-5,max_value=1e-2,sampling="LOG",default=1e-3)),loss=tf.losses.MeanSquaredError(),metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error])


		return full_model

class ModeloCompletoI_Concat_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0.99,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=120,activation='relu'))(model_1_1)
		model_1_1 = tf.keras.layers.Dropout(rate=0)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.54,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=7,activation='relu',padding='valid',return_sequences=True)(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=7,activation='relu',padding='valid',return_sequences=False)(model_3)
		model_3 = tf.keras.layers.MaxPooling2D(pool_size=5)(model_3)
		model_3 = tf.keras.layers.Flatten()(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation='relu')(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)
		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(optimizer=tf.optimizers.Adam(hp.Float("learning_rate",min_value=1e-5,max_value=1e-2,sampling="LOG",default=1e-3)),loss=tf.losses.MeanSquaredError(),metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error])

		return full_model

class ModeloCompletoI_Suma_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0.99,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=120,activation='relu'))(model_1_1)
		model_1_1 = tf.keras.layers.Dropout(rate=0)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.54,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=7,activation='relu',padding='valid',return_sequences=True)(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=7,activation='relu',padding='valid',return_sequences=False)(model_3)
		model_3 = tf.keras.layers.MaxPooling2D(pool_size=5)(model_3)
		model_3 = tf.keras.layers.Flatten()(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation='relu')(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)
		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(optimizer=tf.optimizers.Adam(hp.Float("learning_rate",min_value=1e-5,max_value=1e-2,sampling="LOG",default=1e-3)),loss=tf.losses.MeanSquaredError(),metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error])

		return full_model

class ModeloCompletoII_Concat_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.63,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.03),dropout=0.899,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=384,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.09,return_sequences=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=tf.losses.MeanSquaredError(),
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompletoII_Suma_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.63,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.03),dropout=0.899,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=384,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.09,return_sequences=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])
		
		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=tf.losses.MeanSquaredError(),
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompletoIII_Concat_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0824),dropout=0.899,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.18,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=72,activation='tanh')(model_2_2)
		model_2_2 = tf.keras.layers.Dropout(rate=0.36)(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=5,activation='relu',padding='valid',return_sequences=True)(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=3,activation='relu',padding='same',return_sequences=False)(model_3)
		model_3 = tf.keras.layers.MaxPooling2D(pool_size=5)(model_3)
		model_3 = tf.keras.layers.Flatten()(model_3)
		model_3 = tf.keras.layers.Dense(units=48,activation='sigmoid')(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=tf.losses.MeanSquaredError(),
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompletoIII_Suma_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0824),dropout=0.899,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.18,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=72,activation='tanh')(model_2_2)
		model_2_2 = tf.keras.layers.Dropout(rate=0.36)(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=5,activation='relu',padding='valid',return_sequences=True)(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.ConvLSTM2D(filters=8,kernel_size=3,activation='relu',padding='same',return_sequences=False)(model_3)
		model_3 = tf.keras.layers.MaxPooling2D(pool_size=5)(model_3)
		model_3 = tf.keras.layers.Flatten()(model_3)
		model_3 = tf.keras.layers.Dense(units=48,activation='sigmoid')(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=tf.losses.MeanSquaredError(),
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompletoIV_Concat_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0824),dropout=0.899,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.18,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=72,activation='tanh')(model_2_2)
		model_2_2 = tf.keras.layers.Dropout(rate=0.36)(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=5,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.18,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=72,activation='sigmoid')(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=tf.losses.MeanSquaredError(),
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompletoIV_Suma_Version6(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0824),dropout=0.899,return_sequences=True)(model_1_1)
		model_1_1 = tf.keras.layers.Flatten()(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=24,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.18,return_sequences=False)(input_2)
		model_2_2 = tf.keras.layers.Dense(units=72,activation='tanh')(model_2_2)
		model_2_2 = tf.keras.layers.Dropout(rate=0.36)(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=24,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=5,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0075),dropout=0.18,return_sequences=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=72,activation='sigmoid')(model_3)
		model_3 = tf.keras.layers.Dense(units=24,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=tf.losses.MeanSquaredError(),
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model


class ModeloCompleto_I2_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0975),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.03),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.99,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=120,activation='relu')(model_3)
		model_3 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_I2_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0975),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.03),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0.99,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=120,activation='relu')(model_3)
		model_3 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_I3_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.09,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=384,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0225),dropout=0,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0675),dropout=0.27,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_I3_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.09,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=384,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0225),dropout=0,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='same'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0675),dropout=0.27,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_I5_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.36,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.09),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='relu',padding='valid'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0.99,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_3)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2, model_3])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_I5_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,image_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.image_input_shape = image_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)
		input_3 = tf.keras.layers.Input(shape=self.image_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.36,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.09),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=7,activation='relu',padding='valid'))(input_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=3))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=8,kernel_size=3,activation='relu',padding='valid'))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=5))(model_3)
		model_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(model_3)
		model_3 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0.99,return_sequences=False,stateful=False)(model_3)
		model_3 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_3)

		output = tf.keras.layers.Add()([model_1_1, model_2_2, model_3])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2, input_3], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model


class ModeloCompleto_SinImagen_I2_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0975),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.03),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I2_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0975),dropout=0.27,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.03),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.LSTM(units=320,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Add()([model_1_1, model_2_2])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I3_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.09,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=384,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0225),dropout=0,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I3_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.LSTM(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.09,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.LSTM(units=384,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0225),dropout=0,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Add()([model_1_1, model_2_2])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I5_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.36,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.09),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I5_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.36,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.09),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Add()([model_1_1, model_2_2])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model


class ModeloCompleto_SinImagen_I15_Concat(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0975),dropout=0.54,return_sequences=False)(input_1)
		model_1_1 = tf.keras.layers.Dense(units=72,activation='relu')(model_1_1)
		model_1_1 = tf.keras.layers.Dropout(rate=0)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=120,activation='relu'))(model_2_2)
		model_2_2 = tf.keras.layers.Dropout(rate=0)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Concatenate()([model_1_1, model_2_2])
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_1", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_1_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)
		output = tf.keras.layers.Dense(units=hp.Int("dense_aditional_layer_2", min_value=24, max_value=120, step=24, default=24),activation=hp.Choice("dense_aditional_layer_2_activation",values=["relu", "tanh", "sigmoid"],default="relu"))(output)

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I15_Suma(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.0975),dropout=0.54,return_sequences=False)(input_1)
		model_1_1 = tf.keras.layers.Dense(units=72,activation='relu')(model_1_1)
		model_1_1 = tf.keras.layers.Dropout(rate=0)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_1_1)

		model_2_2 = tf.keras.layers.LSTM(units=64,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.105),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=120,activation='relu'))(model_2_2)
		model_2_2 = tf.keras.layers.Dropout(rate=0)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(model_2_2)

		output = tf.keras.layers.Add()([model_1_1, model_2_2])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model

class ModeloCompleto_SinImagen_I5_Suma_ActivacionDensas(HyperModel):

	def __init__(self,hourly_input_shape,daily_input_shape,n_steps_out):
		self.hourly_input_shape = hourly_input_shape
		self.daily_input_shape = daily_input_shape
		self.n_steps_out = n_steps_out

	def build(self, hp):

		input_1 = tf.keras.layers.Input(shape=self.hourly_input_shape)
		input_2 = tf.keras.layers.Input(shape=self.daily_input_shape)

		model_1_1 = tf.keras.layers.GRU(units=448,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.06),dropout=0.36,return_sequences=True)(input_1)
		model_1_1 = tf.keras.layers.GRU(units=128,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0.09),dropout=0.36,return_sequences=False)(model_1_1)
		model_1_1 = tf.keras.layers.Dense(units=self.n_steps_out,activation='relu')(model_1_1)

		model_2_2 = tf.keras.layers.GRU(units=256,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(input_2)
		model_2_2 = tf.keras.layers.GRU(units=512,activation='tanh',kernel_regularizer=tf.keras.regularizers.L1(l1=0),dropout=0,return_sequences=True)(model_2_2)
		model_2_2 = tf.keras.layers.Flatten()(model_2_2)
		model_2_2 = tf.keras.layers.Dense(units=self.n_steps_out,activation='relu')(model_2_2)

		output = tf.keras.layers.Add()([model_1_1, model_2_2])

		output = tf.keras.layers.Dense(units=self.n_steps_out,activation=None)(output)

		full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

		full_model.compile(
			optimizer=tf.optimizers.Adam(
				hp.Float("learning_rate",
					min_value=1e-5,
					max_value=1e-2,
					sampling="LOG",
					default=1e-3)),
			loss=CustomMetrics.symmetric_mean_absolute_percentage_error,
			metrics=[tf.metrics.MeanAbsoluteError(),tf.keras.metrics.MeanAbsolutePercentageError(),CustomMetrics.symmetric_mean_absolute_percentage_error]
		)

		return full_model