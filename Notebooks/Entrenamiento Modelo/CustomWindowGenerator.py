import numpy as np
import pandas as pd
import tensorflow as tf

"""
Script que se encarga de realizar la construcción de la ventana de datos y retor-
nar el generador de los mismos.
"""

class MyWindowDatasetGenerator():
    def __init__(self, data, times, shift, input_signals, output_signals):
        
        self.times = times
        self.shift = shift
        
        self.column_indices = {name: i for i, name in enumerate(data.columns)}
        
        self.input_signals = input_signals
        self.output_signals = output_signals
        
        
    def split_window(self,features):
        input_slice = slice(0, self.times)
        labels_slice = slice(self.times, None)

        inputs = features[:, input_slice, :]
        labels = features[:, labels_slice, :]
        
        inputs = tf.stack([inputs[:, :, self.column_indices[name]] for name in self.input_signals],axis=-1)
        labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.output_signals],axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.times, None])
        labels.set_shape([None, self.shift, None])

        return inputs, labels
    
    def make_dataset(self, data, batch_size,suffle=False,sequence_stride=1):
    	data = np.array(data, dtype=np.float32)
    	ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.times+self.shift,
            sequence_stride=sequence_stride,
            shuffle= suffle,
            batch_size=batch_size)

    	ds = ds.map(self.split_window)

    	return ds