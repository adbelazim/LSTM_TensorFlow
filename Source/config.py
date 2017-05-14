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


#Loss: 'mean_squared_logarithmic_error', 'mean_squared_error'
def get_compiler_config():
   compiler_config = {'loss' : 'mean_squared_error'}
   return compiler_config

def get_fit_config():
   fit_config = {'epochs' : 5,
   				'batch_size' : 26}
   return fit_config

#Activation functions: 'sigmoid', 'linear', 'tanh', 'hard_sigmoid'
#Kernel_initializer: uniform, VarianceScaling, TruncatedNormal, Orthogonal, lecun_uniform, glorot_normal, glorot_uniform
def get_nn_config():
	nn_config = {'activation_function' : 'tanh',
				'recurrent_activation' : 'hard_sigmoid',
				'kernel_initializer' : 'glorot_uniform',
				'recurrent_initializer' :'orthogonal', 
				'input_dim' : 1,
				'time_steps' : 48,
				'batch_size' : 26,
				'dropout' : 0,
				'recurrent_dropout' : 0,
				'use_bias' : True,
				'unit_forget' : False,
				'go_backwards' : False,
				'units' : 48,
				'output' : 1,
				'layers' : 1
				}
	return nn_config


def get_files_config():
	case = str(sys.argv[1])
	order = str(sys.argv[2])
	file_data = "../Data/"
	filename_train = ""
	if order == "1":
		filename_train = file_data + "1" + case + ".txt"
		filename_test = file_data + "2" + case + ".txt"
	else:
		filename_train = file_data + "2" + case + ".txt"
		filename_test = file_data + "1" + case + ".txt"

	files_config = {'filename_train' : filename_train,
					'filename_test' : filename_test
					}
	return files_config


def get_files_save(units,layers):
	case = str(sys.argv[1])
	order = str(sys.argv[2])
	nn_config = get_nn_config()
	path_dir = "/Users/cristobal/Documents/Tesis/Codigo/Neural_LSTM/Checkpoint/"
	improvements = "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"

	layer = str(layers)
	units = str(units)
	#dir_save se utiliza para crear el directorio a guardar los pesos
	dir_save = path_dir + case + "/" + layer +"_layer"+ "/" + units + "_units"  + "/" + order + "_fold"
	weights_dir = dir_save + "/" + "weights"
	dir_tensor_board =  dir_save + "/" + "tensor_log"
	dir_weights_image = dir_save + "/" + "heat_images"
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)
	if not os.path.exists(weights_dir):
		os.makedirs(weights_dir)
	if not os.path.exists(dir_tensor_board):
		os.makedirs(dir_tensor_board)
	if not os.path.exists(dir_weights_image):
		os.makedirs(dir_weights_image)

	filepath = dir_save + "/" +"weights/"+ improvements
	file_csv = dir_save + "/" + "training_log.csv"
	file_tensor_board = dir_save + "/" + "tensor_log"
	file_weights = dir_save + "/" + "weights/" + "final_weights.hdf5"

	files_save = { 'file_save' : dir_save,
					'filepath' : filepath,
					'file_csv' : file_csv,
					'file_weights' : file_weights,
					'file_tensor_board' : file_tensor_board,
					'dir_weights_image' : dir_weights_image,
					'period' : 10

	}
	return files_save











