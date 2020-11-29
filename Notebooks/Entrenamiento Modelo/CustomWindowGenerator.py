import numpy as np
import pandas as pd

"""
Script que se encarga de realizar la construcción de la ventana de datos y retor-
nar el generador de los mismos.
"""

class WindowGenerator():
	"""
	Clase encarga de implementar los métodos de construcción de la ventana de datos 
	y del generador.
	Inputs:
		- samples: Entero que representa la cantidad de muestras que contendrá la  ventana
		de datos.
		- times: Número entero de tiempos de la ventana de datos.
		- shift: Entero que representa el desplazamiento de horas involucrado en la venta-
		na de datos.
		- input_columns: Lista con las series que estarán incluidas en la ventana de datos.
		- output_columns: Lista con las series que quieren predecir.
		- column_indices: Diccionario que contiene el indicador de número de columna dentro
		del dataframe para cada una de las series.
	"""
    def __init__(self,samples,times,shift,inputs_columns,output_columns,column_indices):

    	self.samples = samples
    	self.times = times
    	self.shift = shift
    	self.inputs_columns = inputs_columns
    	self.output_columns = output_columns

    	self.id_x_signals = [column_indices.get(key) for key in inputs_columns]
    	self.id_y_signals = [column_indices.get(key) for key in output_columns]


    def data_build(self,df):
    	"""
    	Este método se encarga de tomar un Dataframe y retonar el Tensor de datos que re-
    	presenta la serie. Para realizar esto, inicialmente se calcula el número de seri-
    	es predictoras y series a predecir, luego se construye para cada uno de     estos
    	conjuntos un tensor de ceros con las dimensiones (samples,times,series).   Poste-
    	riormente con un ciclo que empieza en 0 y termina en el número muestras de     la
    	ventana se extraeran la cantidad de filas correspondiente al numero de    tiempos
    	para las series predictoras y para las series a predecir se extraera la  cantidad
    	correpondiente al número de desplazamientos.
    	Inputs:
    		- df: Dataframe con los datos.
    	Outputs:
    		- Tupla de Tensores con dimensiones ((samples,times,x_signals),(samples,times,y_signals))
    	"""
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
    	"""
    	Esta función se encarga de construir batches de datos. Para realizar esto el méto-
    	do tiene dos opciones particulares, cuando los batches son aleatorios y cuando no.
    	En el primer caso por medio de un ciclo for que va desde 0 hasta batches se genera
    	en cada iteración un número aleatorio entre 0 y samples, dicho valor será un indi-
    	ce que permitira extraer una serie particular del tensor de datos, tanto para    X
    	cómo para Y. En el segundo caso se genera un número aleagorio entre 0 y   samples-
    	batches, con dicho valor se generará un ciclo for que irá desde él mismo hasta  él
    	más número de batches y en cada iteración agregara la serie correspondiente.
    	Inputs:
    		- x: Tensor de datos de series predictoras.
    		- y: Tensor de datos de series a predecir.
    		- batches: Entero que determina el número de batches a extraer.
    		- shuffle: Booleano que indica si los batches serán aleatorios o no.
    	Output:
    		- generator: Generador que por cada llamado entrega un batch de datos.
    	"""
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