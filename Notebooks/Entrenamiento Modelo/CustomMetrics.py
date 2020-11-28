import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def symmetric_mean_absolute_percentage_error(y_true,y_pred):
	#https://gist.github.com/arnaldog12/5f2728f229a8bd3b4673b72786913252#file-custom_metrics-py
    '''Calculates the symmetric mean absolute percentage error (smape)
    rate between predicted and target values.
    '''
    sum_numerator   = 2 * K.abs(y_pred-y_true)
    sum_denominator = K.clip(K.abs(y_true)+K.abs(y_pred), K.epsilon(), np.inf)
    sum_factor = K.sum(sum_numerator/sum_denominator)
    sum_factor = tf.cast(sum_factor,dtype=tf.float64)
    proportion_factor = 100 / len(y_true)
    
    return proportion_factor * sum_factor