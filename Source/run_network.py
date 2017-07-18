import data as dt
import model_fit
import model_lstm as m_lstm
import metrics
import save_model as save
import config as cfg
import step_test
from keras.utils import plot_model
from mds_visualization import collapse_layer
import numpy as np

import grid_search

#TO DO
#import model_lstm_cnn as m_lstm_cnn
#import model_cnn as m_cnn
#import model_distributed as m_dist

def run_network(subject,
				units, 
				layers, 
				case, 
				order, 
				collapse_matrix_2gates,
				collapse_matrix_2gates_flat,
				collapse_matrix_2gates_mg,
				collapse_matrix_3gates,
				collapse_matrix_3gates_flat,
				collapse_matrix_3gates_mg,
				collapse_matrix_4gates,
				collapse_matrix_4gates_flat,
				collapse_matrix_4gates_mg,
				label_matrix, stateful = False,sampling_time=0.2):

	
	grid_config = cfg.get_grid_config()

	#Se cargan los datos de entrenamiento y test
	#trainX, testX, trainY, testY, sampling_time,scaler_cbfv, scaler_abp = dt.train_test_data(case,order)

	#callbacks = model_fit.model_fit_callbacks(subject,units,layers,case,order)

	#Se compila el modelo
	#model = m_lstm.init_model_lstm(units,layers) 
	#model = m_lstm.init_model_lstm_for_search(units,layers) 
	

	model = grid_search.grid_search()
	print(model.get_config())

	trainX, testX, trainY, testY, sampling_time,scaler_cbfv, scaler_abp = dt.train_test_data()

	#Se entrena el modelo
	#model = model_fit.model_fit(model,trainX, trainY, testX, testY,subject,units,layers,case,order,stateful = stateful)

	if stateful == False:
		#Se predice como se ajustan los conjuntos de entrenamiento y test al modelo
		train_predict = model.predict(trainX)
   		test_predict = model.predict(testX)

	else:
		fit_config = cfg.get_fit_config()
		# make predictions with batch_size
   		train_predict = model.predict(trainX,batch_size=fit_config['batch_size'])
   		test_predict = model.predict(testX,batch_size=fit_config['batch_size'])

   	#se predice la respuesta al escalon

   	step_test.step_prediction(trainX[:,0,0],sampling_time,model,grid_config['path_subjects'],subject,units,layers,case,order,'step_test_normalize.png')
 
	#Se obtiene el valor no normalizado de las predicciones. Ivert predictions
   	train_predict = scaler_cbfv.inverse_transform(train_predict)
   	trainY = scaler_cbfv.inverse_transform([trainY])
   	test_predict = scaler_cbfv.inverse_transform(test_predict)
   	testY = scaler_cbfv.inverse_transform([testY])

   	#Se calcula la correlacion
   	corr_train, corr_test = metrics.correlations(train_predict,trainY,test_predict,testY)

   	print "Corr_train: " + str(corr_train[0,1]) + " " + "Corr_test: " + str(corr_test[0,1])

   	#se acepta solamente los modelos que tengan al menos 70% de correlacion en train y test
   	if corr_train[0,1] >= 0.7 and corr_test[0,1] >= 0.7:
	   	
	  	collapse_layer(collapse_matrix_2gates,model.layers[1].get_weights()[1],units,mean_method='aritmetic',gates_number = 2)
	  	collapse_layer(collapse_matrix_2gates_flat,model.layers[1].get_weights()[1],units,mean_method='flatten',gates_number = 2)
	  	collapse_layer(collapse_matrix_2gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric',gates_number = 2)

	  	collapse_layer(collapse_matrix_3gates,model.layers[1].get_weights()[1],units,mean_method='aritmetic',gates_number = 3)
	  	collapse_layer(collapse_matrix_3gates_flat,model.layers[1].get_weights()[1],units,mean_method='flatten',gates_number = 3)
	  	collapse_layer(collapse_matrix_3gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric',gates_number = 3)

	  	collapse_layer(collapse_matrix_4gates,model.layers[1].get_weights()[1],units,mean_method='aritmetic',gates_number = 4)
	  	collapse_layer(collapse_matrix_4gates_flat,model.layers[1].get_weights()[1],units,mean_method='flatten',gates_number = 4)
	  	collapse_layer(collapse_matrix_4gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric',gates_number = 4)
			
		label_recurrent_kernel = case + "_recurrent_kernel"
		label_matrix.append(label_recurrent_kernel)

	collapse_layer(collapse_matrix_2gates,model.layers[1].get_weights()[1],units,mean_method='aritmetic',gates_number = 2)
	collapse_layer(collapse_matrix_2gates_flat,model.layers[1].get_weights()[1],units,mean_method='flatten',gates_number = 2)
	collapse_layer(collapse_matrix_2gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric',gates_number = 2)

	collapse_layer(collapse_matrix_3gates,model.layers[1].get_weights()[1],units,mean_method='aritmetic',gates_number = 3)
	collapse_layer(collapse_matrix_3gates_flat,model.layers[1].get_weights()[1],units,mean_method='flatten',gates_number = 3)
	collapse_layer(collapse_matrix_3gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric',gates_number = 3)

	collapse_layer(collapse_matrix_4gates,model.layers[1].get_weights()[1],units,mean_method='aritmetic',gates_number = 4)
	collapse_layer(collapse_matrix_4gates_flat,model.layers[1].get_weights()[1],units,mean_method='flatten',gates_number = 4)
	collapse_layer(collapse_matrix_4gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric',gates_number = 4)
			
	label_recurrent_kernel = case + "_recurrent_kernel"
	label_matrix.append(label_recurrent_kernel)

   	save.save_model(model,subject,units,layers,case,order)
   	save.save_txt(corr_train,corr_test,subject,units,layers,case,order)
   	save.plot_model_predictions(train_predict,trainY,test_predict,testY,corr_train,corr_test,subject,units,layers,case,order)

   	del model















