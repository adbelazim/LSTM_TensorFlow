import numpy as np
np.random.seed(997)
from keras import initializers
seed_value = 997


def get_nn_config():
	kernel_initializer = initializers.glorot_uniform(seed=seed_value)
	recurrent_initializer = initializers.orthogonal(seed=seed_value)
	nn_config = {'activation_function' : 'tanh',
				'recurrent_activation' : 'hard_sigmoid',
				'kernel_initializer' : kernel_initializer,
				'recurrent_initializer' :recurrent_initializer, 
				'input_dim' : 1,
				'time_steps' : 10,
				'batch_size' : 3,
				'dropout' : 0,
				'recurrent_dropout' : 0,
				'use_bias' : True,
				'unit_forget' : True,
				'go_backwards' : False,
				'units' : 48,
				'output' : 1,
				'layers' : 1
				}
	return nn_config