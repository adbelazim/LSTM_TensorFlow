from keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(997)

from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import LambdaCallback
from keras.callbacks import CSVLogger
from keras.callbacks import History
from keras.callbacks import TensorBoard

import callbacks_model as callb
import config as cfg
import config_keras as cfg_keras



# callbacks uses for fit de model 
def model_fit_callbacks(subject,units,layers,case,order):

	nn_config = cfg_keras.get_nn_config()
	files_save = cfg.get_files_save(subject,units,layers,case,order,nn_config)

	checkpointer = ModelCheckpoint(filepath=files_save['filepath'], 
									verbose=1, 
									save_best_only=True,
									save_weights_only=True, 
									period=files_save['period'])

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
								factor=0.2, 
								patience=10, 
								min_lr=0)

	early_stoping = EarlyStopping(monitor='val_loss', 
								min_delta=0.001, 
								patience=15, 
								verbose=0, 
								mode='auto')

	loss_history = callb.LossHistory()
	csv_logger = CSVLogger(files_save['file_csv'], separator=',', append=False)
	fit_history = History()
	tensor_board = TensorBoard(log_dir=files_save['file_tensor_board'], histogram_freq=0, write_graph=True, write_images=True)
	return [checkpointer,reduce_lr,early_stoping,loss_history,fit_history,csv_logger,tensor_board]
	#return [checkpointer,loss_history,fit_history,csv_logger,tensor_board]


def model_fit(model,trainX, trainY, testX, testY,subject,units,layers,case,order,stateful = False):

	fit_config = cfg.get_fit_config()
	callbacks = model_fit_callbacks(subject,units,layers,case,order)

	if stateful:
		model.fit(trainX, trainY, 
			epochs=fit_config['epochs'], 
			batch_size=fit_config['batch_size'], 
			verbose=0,
			validation_data=(testX,testY), 
			callbacks=callbacks,
			shuffle=False)
	else:
		model.fit(trainX, trainY, 
			epochs=fit_config['epochs'], 
			verbose=0,
			validation_data=(testX,testY), 
			callbacks=callbacks)
		#print(callbacks[3].losses)

	return model

def model_fit_for_search(model,trainX, trainY, testX, testY,units,layers,stateful = False):
	import global_queue
	import config as cfg

	subject = global_queue.get_actual_subject(global_queue.subjects_list)
	case = global_queue.get_actual_case(global_queue.cases_list)
	order = global_queue.get_actual_order(global_queue.orders_list)

	fit_config = cfg.get_fit_config()
	callbacks = model_fit_callbacks(subject,units,layers,case,order)

	if stateful:
		model.fit(trainX, trainY, 
			epochs=fit_config['epochs'], 
			batch_size=fit_config['batch_size'], 
			verbose=2,
			validation_data=(testX,testY), 
			callbacks=callbacks,
			shuffle=False)
	else:
		model.fit(trainX, trainY, 
			epochs=fit_config['epochs'], 
			verbose=2,
			validation_data=(testX,testY), 
			callbacks=callbacks)
		#print(callbacks[3].losses)

	return model




