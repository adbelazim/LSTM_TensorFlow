from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Conv1D
from keras import optimizers
from keras import regularizers

import config as cfg
import time

def init_model_lstm_cnn(activation_function, 
						recurrent_activation,
						kernel_initializer, 
						recurrent_initializer, 
						dropout, 
						epochs, 
						recurrent_dropout, 
						unit_forget, 
						use_bias, 
						loss,
						go_backwards):

   model = Sequential()  
  
   model.add(LSTM(time_steps,input_shape=(time_steps,input_dim),
		activation = activation_function,
        go_backwards = go_backwards,
        recurrent_activation = recurrent_activation, 
        return_sequences = True, 
		use_bias = use_bias, 
		kernel_initializer = kernel_initializer, 
		recurrent_initializer = recurrent_initializer,
		unit_forget_bias = unit_forget,
		recurrent_dropout = recurrent_dropout,
		kernel_regularizer=regularizers.l2(l2),  
        activity_regularizer=regularizers.l1(l1)))
   
   model.add(Conv1D(filters = 1, 
		kernel_size = 1, 
		strides=1, 
		padding='valid', 
		dilation_rate=1, 
		activation=activation_function, 
		use_bias=True, 
		kernel_initializer=kernel_initializer, 
		bias_initializer='zeros', 
		kernel_regularizer=None, 
		bias_regularizer=None, 
		activity_regularizer=None, 
		kernel_constraint=None, 
		bias_constraint=None))
   
   model.add(Flatten()) 
   
   model.add(Dense(1,
		activation = activation_function,
		use_bias = True))

   optimizer_config = cfg.get_optimizer_config()
   optimizer = optimizers.RMSprop(lr=optimizer_config['lr'],
                                  rho =optimizer_config['rho'] ,
                                  epsilon=optimizer_config['epsilon'],
                                  decay=optimizer_config['decay'])
   start = time.time()
   #model.compile(loss=loss, optimizer=optimizer,metrics=['mae'])  
   compiler_config = cfg.get_compiler_config()
   model.compile(loss=compiler_config['loss'], optimizer=optimizer)  
   print "Compilation Time : ", time.time() - start
   return model




