from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

import config as cfg
import model_lstm as m_lstm
import model_fit
from data import train_test_data
import config_keras as cfg_keras

import global_queue
from data_utils import lectura, lineal_interpolation, create_timesteps

def model_lstm():

	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import LSTM
	from keras import regularizers
	import config as cfg
	import config_keras as cfg_keras
	from data import train_test_data
	import time
	import metrics
	import global_queue

	trainX, testX, trainY, testY, sampling_time,scaler_cbfv, scaler_abp = train_test_data()

	nn_config = cfg_keras.get_nn_config()
	regularizer_config = cfg.get_regularizer_config()
	stateful = False

	#Kernel_initializer: uniform, VarianceScaling, TruncatedNormal, Orthogonal, lecun_uniform, glorot_normal, glorot_uniform

	model = Sequential()  
	return_sequences = True
	for _ in range(nn_config['layers']):
		if _ == nn_config['layers'] - 1:
			return_sequences = False
		model.add(LSTM(nn_config['units'],input_shape=(nn_config['time_steps'],nn_config['input_dim']),
		activation = {{choice(['hard_sigmoid', 'sigmoid','tanh','linear'])}},
		go_backwards = nn_config['go_backwards'],
		recurrent_activation = {{choice(['hard_sigmoid', 'sigmoid','tanh','linear'])}}, 
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

	start = time.time()
	compiler_config = cfg.get_compiler_config()
	model.compile(loss=compiler_config['loss'], optimizer={{choice(['rmsprop','adam'])}})  

	
	print "Compilation Time : ", time.time() - start

	model = model_fit.model_fit_for_search(model,trainX, trainY, testX, testY,nn_config['units'],nn_config['layers'],stateful)
	if stateful == False:
		#Se predice como se ajustan los conjuntos de entrenamiento y test al modelo
		train_predict = model.predict(trainX)
   		test_predict = model.predict(testX)

	else:
		fit_config = cfg.get_fit_config()
		# make predictions with batch_size
   		train_predict = model.predict(trainX,batch_size=fit_config['batch_size'])
   		test_predict = model.predict(testX,batch_size=fit_config['batch_size'])

	train_predict = scaler_cbfv.inverse_transform(train_predict)
   	trainY = scaler_cbfv.inverse_transform([trainY])
   	test_predict = scaler_cbfv.inverse_transform(test_predict)
   	testY = scaler_cbfv.inverse_transform([testY])

   	#Se calcula la correlacion
   	corr_train, corr_test = metrics.correlations(train_predict,trainY,test_predict,testY)
	print('Test corr:', corr_test[0,1])
	return {'loss': -corr_test[0,1], 'status': STATUS_OK, 'model': model}

   


def grid_search():
	grid_config = cfg.get_grid_config()

	trainX, testX, trainY, testY, sampling_time,scaler_cbfv, scaler_abp = train_test_data()

	print("comienza grid search")
	best_run, best_model = optim.minimize(model=model_lstm,
										data=train_test_data,
										algo=tpe.suggest,
										max_evals=15,
										verbose = True,
										trials=Trials())
	print("fin grid search")

	#print("Evalutation of best performing model:")
	#print(best_model.evaluate(testX, testY))
	#print("Best performing model chosen hyper-parameters:")
	#print(best_run)

	return best_model














