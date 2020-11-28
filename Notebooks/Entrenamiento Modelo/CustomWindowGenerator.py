import numpy as np
import pandas as pd

class WindowGenerator():
    def __init__(self,samples,times,shift,inputs_columns,output_columns,column_indices):

    	self.samples = samples
    	self.times = times
    	self.shift = shift
    	self.inputs_columns = inputs_columns
    	self.output_columns = output_columns

    	self.id_x_signals = [column_indices.get(key) for key in inputs_columns]
    	self.id_y_signals = [column_indices.get(key) for key in output_columns]


    def data_build(self,df):
    
	    num_x_signals = len(self.inputs_columns)
	    num_y_signals = len(self.output_columns)
	        
	    x_shape = (self.samples, self.times, num_x_signals)
	    x_batch = np.zeros(shape=x_shape, dtype=np.float16)

	    y_shape = (self.samples, self.shift, num_y_signals)
	    y_batch = np.zeros(shape=y_shape, dtype=np.float16)

	    for idx in range(self.samples):
	        x_batch[idx] = df.iloc[idx:idx+self.times,self.id_x_signals].values
	        y_batch[idx] = df.iloc[idx+self.times:idx+self.times+self.shift,self.id_y_signals].values
	        
	    return (x_batch,y_batch)

    def batch_generator(self,x,y,batches,shuffle=True):

    	while True:
	        x_batch = np.zeros(shape=(batches,self.times,len(self.inputs_columns)), dtype=np.float16)
	        y_batch = np.zeros(shape=(batches,self.times,len(self.output_columns)), dtype=np.float16)

	        if shuffle:
	            for i in range(batches):
	                idx = np.random.randint(self.samples)
	                x_batch[i] = x[idx]
	                y_batch[i] = y[idx]
	        else:
	            idx = np.random.randint(self.samples-batches)
	            batch_idx = 0
	            for i in range(idx,idx+batches,1):
	                x_batch[batch_idx] = x[i]
	                y_batch[batch_idx] = y[i]
	                batch_idx += 1

	        yield (x_batch, y_batch)