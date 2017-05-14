from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from keras import regularizers

#from keras.layers import Bidirectional
import config as cfg
import metrics
import time

from tensorflow import losses
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_pearson_correlation as pearson


def init_model_lstm_stateful(units,layers):

   nn_config = cfg.get_nn_config()
   regularizer_config = cfg.get_regularizer_config()

   #Se deberia llamar algun archivo de configuracion con los parametros de la funcion para no recibirlos de entrada
   model = Sequential()  
   
   #model.add(LSTM(look_back*mult_cell,batch_input_shape=(batch_size,time_steps,look_back),
   model.add(LSTM(units, batch_input_shape=(nn_config['batch_size'],nn_config['time_steps'],nn_config['input_dim']),
        activation = nn_config['activation_function'],
        go_backwards = nn_config['go_backwards'],
        recurrent_activation = nn_config['recurrent_activation'], 
        return_sequences = True, 
      	use_bias = nn_config['use_bias'], 
      	kernel_initializer = nn_config['kernel_initializer'], 
      	recurrent_initializer = nn_config['recurrent_initializer'],
      	unit_forget_bias = nn_config['unit_forget'],
      	recurrent_dropout = nn_config['recurrent_dropout'],
      	kernel_regularizer=regularizers.l2(regularizer_config['l2']),  
        activity_regularizer=regularizers.l1(regularizer_config['l1']),
	      stateful=True))
  

   model.add(LSTM(units/2,batch_input_shape=(nn_config['batch_size'],nn_config['time_steps'],nn_config['input_dim']),
	      activation = nn_config['activation_function'],
        go_backwards = nn_config['go_backwards'],
        recurrent_activation = nn_config['recurrent_activation'], 
        return_sequences = False, 
      	use_bias = nn_config['use_bias'], 
      	kernel_initializer = nn_config['kernel_initializer'], 
      	recurrent_initializer = nn_config['recurrent_initializer'],
      	unit_forget_bias = nn_config['unit_forget'],
      	recurrent_dropout = nn_config['recurrent_dropout'],
      	stateful=True))

   
   model.add(Dense(1,
      	activation = nn_config['activation_function'],
      	use_bias = nn_config['use_bias']))
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

def init_model_lstm(units,layers):

   nn_config = cfg.get_nn_config()
   regularizer_config = cfg.get_regularizer_config()

   model = Sequential()  
   return_sequences = True
   for _ in range(layers):
        print(_)
        if _ == nn_config['layers'] - 1:
          return_sequences = False
        model.add(LSTM(units,input_shape=(nn_config['time_steps'],nn_config['input_dim']),
        #model.add(LSTM(nn_config['units'],input_shape=(nn_config['input_dim'],nn_config['time_steps']),
        activation = nn_config['activation_function'],
        go_backwards = nn_config['go_backwards'],
        recurrent_activation = nn_config['recurrent_activation'], 
        return_sequences = return_sequences, 
        use_bias = nn_config['use_bias'], 
        kernel_initializer = nn_config['kernel_initializer'], 
        recurrent_initializer = nn_config['recurrent_initializer'],
        unit_forget_bias = nn_config['unit_forget'],
        recurrent_dropout = nn_config['recurrent_dropout'],
        kernel_regularizer=regularizers.l2(regularizer_config['l2']),  
        activity_regularizer=regularizers.l2(regularizer_config['l1'])))
  
   model.add(Dense(nn_config['output'],
	      activation = nn_config['activation_function'],
	      use_bias = nn_config['use_bias']))

   optimizer_config = cfg.get_optimizer_config()
   optimizer = optimizers.RMSprop(lr=optimizer_config['lr'],
                                  rho =optimizer_config['rho'] ,
                                  epsilon=optimizer_config['epsilon'],
                                  decay=optimizer_config['decay'])

   start = time.time()
   compiler_config = cfg.get_compiler_config()
   model.compile(loss=compiler_config['loss'], optimizer=optimizer)  
   #model.compile(loss=losses.log_loss, optimizer=optimizer)  
   print "Compilation Time : ", time.time() - start

   return model





