from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras import optimizers
from keras import regularizers

###TO DO
def init_model_distributed_fnn(activation_function, kernel_initializer, dropout, epochs, unit_forget, use_bias, loss):
   
   model = Sequential()

   model.add(TimeDistributed(Dense(units,
	input_shape=(time_steps,look_back),
	activation = activation_function
	)))
  
   model.add(Dnse(1))
 
   optimizer = optimizers.RMSprop(lr=lr,rho =rho ,epsilon=epsilon,decay=decay )
   start = time.time()
   #model.compile(loss=loss, optimizer=optimizer,metrics=['mae'])  
   model.compile(loss=loss, optimizer=optimizer)  
   print "Compilation Time : ", time.time() - start
   return model