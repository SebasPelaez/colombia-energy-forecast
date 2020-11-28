import tensorflow as tf

class CustomRNN(tf.keras.models.Model):
    
    def __init__(self,rnn_units,output_units):
        super(CustomRNN, self).__init__()
        self.rnn_units = rnn_units
        self.output_units = output_units
        
    def build(self, input_shape):

        self.simple_rnn = tf.keras.layers.SimpleRNN(self.rnn_units,
                                               activation='tanh',
                                               return_sequences=False)

        self.output_dense = tf.keras.layers.Dense(units=self.output_units)#,activation='sigmoid')
        
    def call(self, inputs, training=None):

        rnn = self.simple_rnn(inputs)        
        output = self.output_dense(rnn)

        return output

class CustomLSTM(tf.keras.models.Model):
    # TimeDistributed es para la salida de la capa LSTM y poderla combinar con la Densa 
    def __init__(self,lstm_units,output_units):
        super(CustomLSTM, self).__init__()
        self.lstm_units = lstm_units
        self.output_units = output_units
        
    def build(self, input_shape):
        
        self.lstm = tf.keras.layers.LSTM(units=self.lstm_units, 
                                         activation='tanh',
                                         return_sequences=False)
        
        self.output_dense = tf.keras.layers.Dense(units=self.output_units)#,activation='sigmoid')
        
    def call(self, inputs, training=None):
        lstm = self.lstm(inputs)
        output = self.output_dense(lstm)
        
        return output

class CustomGRU(tf.keras.models.Model):
    
    def __init__(self,gru_units,output_units):
        super(CustomGRU, self).__init__()
        self.gru_units = gru_units
        self.output_units = output_units
        
    def build(self, input_shape):
        
        self.gru = tf.keras.layers.GRU(units=self.gru_units, 
                                        activation='tanh',
                                        return_sequences=False)
        
        self.output_dense = tf.keras.layers.Dense(units=self.output_units)#,activation='sigmoid')
        
    def call(self, inputs, training=None):
        gru = self.gru(inputs)
        output = self.output_dense(gru)
        
        return output

class CustomCNN(tf.keras.models.Model):
    
    def __init__(self,dropout_rate,output_units):
        super(CustomCNN, self).__init__()

        self.dropout_rate = dropout_rate
        self.output_units = output_units
        
    def build(self, input_shape):

        self.conv1 = tf.keras.layers.Conv1D(filters=64,kernel_size=2,strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2,padding='valid')
                
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling1D()    
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.output_dense = tf.keras.layers.Dense(units = self.output_units)#,activation='sigmoid')
        
    def call(self, inputs, training=None):

        first_block = self.conv1(inputs)
        first_block = self.bn1(first_block, training=training)
        first_block = self.activation1(first_block)
        first_block = self.pool1(first_block)
        
        global_pooling = self.global_average_pool(first_block)
        dropout_reg = self.dropout(global_pooling, training=training)
        output = self.output_dense(dropout_reg)
        
        return output