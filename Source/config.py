import sys
import os

#Este puede ser, 'adam', 'sgd', 'rmsprop', adadelta, adamax, nadam, TFOptimizer
#optimizer='rmsprop'
def get_optimizer_config():
   optimizer_config = {'lr' : 0.01,
   				'rho' : 0.9,
   				'epsilon' : 1e-08,
   				'decay' : 0.0
   				}
   return optimizer_config

def get_regularizer_config():
   regularizer_config = {'l1' : 0.001,
   				'l2' : 0.001,
   				'epsilon' : 1e-08,
   				'decay' : 0.0
   				}
   return regularizer_config

def get_grid_config():
	grid_config = {'path_subjects' : "../Subjects_test/"}
	return grid_config

#Loss: 'mean_squared_logarithmic_error', 'mean_squared_error'
def get_compiler_config():
   compiler_config = {'loss' : 'mean_squared_error'}
   return compiler_config

def get_fit_config():
   fit_config = {'epochs' : 100,
   				'batch_size' : 3}
   return fit_config

def get_files_config(path_subjects,subject,case):
	file_data = str(path_subjects) + str(subject) + "/"
	
	filename_train = file_data + "1" + str(case) + ".txt"
	filename_test = file_data + "2" + str(case) + ".txt"

	print(subject)
	print(filename_train)
	print(filename_test)

	files_config = {'filename_train' : filename_train,
					'filename_test' : filename_test
					}
	return files_config

def get_files_save(subject,units,layers,case,order,nn_config):
	#nn_config = get_nn_config()
	path = os.getcwd()
	path, last_dir  = os.path.split(path)
	path = path + "/Checkpoint"

	path_dir = path + "/Base_Model/" + subject
	improvements = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

	layer = str(layers)
	units = str(units)
	#dir_save se utiliza para crear el directorio a guardar los pesos
	path_dir = path_dir + "/" +str(nn_config['time_steps']) + "_time_steps" + "/"
	dir_save = path_dir + case + "/" + layer +"_layer"+ "/" + units + "_units"  + "/" + order + "_fold"
	weights_dir = dir_save + "/" + "weights"
	dir_tensor_board =  dir_save + "/" + "tensor_log"
	dir_weights_image = dir_save + "/" + "heat_images"
	dir_dimensionality_reduction = dir_save + "/" + "dimensionality_reduction"

	
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	if not os.path.exists(weights_dir):
		os.makedirs(weights_dir)
	if not os.path.exists(dir_tensor_board):
		os.makedirs(dir_tensor_board)
	if not os.path.exists(dir_weights_image):
		os.makedirs(dir_weights_image)
	if not os.path.exists(dir_dimensionality_reduction):
		os.makedirs(dir_dimensionality_reduction)

	filepath = dir_save + "/" +"weights/"+ improvements
	file_csv = dir_save + "/" + "training_log.csv"
	file_tensor_board = dir_save + "/" + "tensor_log"
	file_weights = dir_save + "/" + "weights/" + "final_weights.hdf5"
	file_txt = dir_save + "/" + "model.txt"
	file_step_txt = dir_save + "/" + "step.txt"

	files_save = { 'file_save' : dir_save,
					'file_txt' : file_txt,
					'filepath' : filepath,
					'file_csv' : file_csv,
					'file_step_txt' : file_step_txt,
					'file_weights' : file_weights,
					'file_tensor_board' : file_tensor_board,
					'dir_weights_image' : dir_weights_image,
					'dir_dimensionality_reduction' : dir_dimensionality_reduction,
					'period' : 100

	}
	return files_save


def get_iterables():
	cases = {'19' : '19',
		'55' : '55',
		'91' : '91'}
	layers = {'1_layer' : '1_layer',
			'2_layer' : '2_layer'}

	cells = {'1_units' : '1_units',
		'2_units' : '2_units',
		'4_units' : '4_units',
		'8_units' : '8_units',
		'16_units' : '16_units',
		'24_units' : '24_units',
		'32_units' : '32_units',
		'40_units' : '40_units',
		'48_units' : '48_units'
		}

	folds = {'1_fold' : '1_fold',
		'2_fold' : '2_fold'}

	return cases, layers, cells, folds








