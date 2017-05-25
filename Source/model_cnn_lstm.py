from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import regularizers

#from keras.layers import Bidirectional
import config as cfg
import metrics
import time

####TODO

def init_model_lstm_stateful(units,layers):

	nn_config = cfg.get_nn_config()
   	regularizer_config = cfg.get_regularizer_config()

	model = Sequential()
	model.add(Dropout(0.25))
	model.add(Conv1D(filters,
	                 kernel_size,
	                 padding='valid',
	                 activation='relu',
	                 strides=1))
	model.add(MaxPooling1D(pool_size=pool_size))
	model.add(LSTM(lstm_output_size))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
