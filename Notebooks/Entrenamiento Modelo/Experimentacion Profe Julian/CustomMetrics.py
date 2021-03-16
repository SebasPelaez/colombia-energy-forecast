import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K

def symmetric_mean_absolute_percentage_error(y_true,y_pred):
    """
    Se calcula la métrica Symmetric Mean Absolute Percentage Error (sMAPE)
    entre los valores predichos y los valores reales.
    El cálcula de esta métrica fue posible gracias a la respuesta propuesta
    en el siguiente enlace:
    https://gist.github.com/arnaldog12/5f2728f229a8bd3b4673b72786913252#file-custom_metrics-py
    Input:
        - y_true: Valores objetivo.
        - y_pred: Valores predichos.
    Output:
        - sMAPE: Métrica calculada.
    """
    sum_numerator   = K.abs(y_pred-y_true)
    sum_denominator = K.maximum(K.abs(y_true)+K.abs(y_pred), K.epsilon())
    sum_factor = K.sum(sum_numerator/sum_denominator)
    sum_factor = tf.cast(sum_factor,dtype=tf.float64)
    proportion_factor = (2 / len(y_true)) * 100.
    
    return proportion_factor * sum_factor