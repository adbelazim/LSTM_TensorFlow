import data as dt
import model_fit
import model_lstm as m_lstm
import metrics
import save_model as save
import config as cfg
import step_test
from keras.utils import plot_model
from mds_visualization import collapse_layer_2gates,collapse_layer_3gates

#TO DO
#import model_lstm_cnn as m_lstm_cnn
#import model_cnn as m_cnn
#import model_distributed as m_dist

def run_network(units, 
				layers, 
				case, 
				order, 
				collapse_matrix_2gates,
				collapse_matrix_2gates_flat,
				collapse_matrix_2gates_mg,
				collapse_matrix_3gates,
				collapse_matrix_3gates_flat,
				collapse_matrix_3gates_mg,
				label_matrix, stateful = False):

	#Se cargan los datos de entrenamiento y test
	trainX, testX, trainY, testY, scaler_cbfv, scaler_abp = dt.train_test_data(case,order,reshape_option = 1)
	print(trainX.shape)
	#Se compila el modelo
	model = m_lstm.init_model_lstm(units,layers) 
	#Se entrena el modelo
	model_fit.model_fit(model,trainX, trainY, testX, testY,units,layers,case,order,stateful = stateful)

	#layer_number = 0
	#for layer in model.layers:
	#	if layer_number < layers:
	#		collapse_layer(collapse_matrix,layer.get_weights()[0],layer_number,units)

	#		label_kernel = case + "_kernel"
	#		label_matrix.append(label_kernel)
			
	#		collapse_layer(collapse_matrix,layer.get_weights()[1],layer_number,units)
			
	#		label_recurrent_kernel = case + "_recurrent_kernel"
	#		label_matrix.append(label_recurrent_kernel)
			
	#		layer_number+=1

	
	if stateful == False:
		#Se predice como se ajustan los conjuntos de entrenamiento y test al modelo
		trainPredict = model.predict(trainX)
   		testPredict = model.predict(testX)

	else:
		fit_config = cfg.get_fit_config()
		# make predictions with batch_size
   		trainPredict = model.predict(trainX,batch_size=fit_config['batch_size'])
   		testPredict = model.predict(testX,batch_size=fit_config['batch_size'])


   	#Values for step test
   	sampling_time = 0.4
   	time = trainX.shape[0]*0.4
   	time_after = time*0.7
   	time_until = time - time_after

   	step = step_test.generate_step(sampling_time = sampling_time, 
   							time_until_release=time_until ,
   							time_after_release = time_after, 
   							smoth_step_stimulus = True)
   	step_prediction = step_test.step_test(model,step)
   	step_test.plot_step_test(step_prediction,units,layers,case,order,'step_test_normalize.png')
   	step_prediction = scaler_cbfv.inverse_transform(step_prediction)
   	step_test.plot_step_test(step_prediction,units,layers,case,order,'step_test.png')


	#Se obtiene el valor no normalizado de las predicciones. Ivert predictions
   	trainPredict = scaler_cbfv.inverse_transform(trainPredict)
   	trainY = scaler_cbfv.inverse_transform([trainY])
   	testPredict = scaler_cbfv.inverse_transform(testPredict)
   	testY = scaler_cbfv.inverse_transform([testY])

   	#Se calcula la correlacion
   	corr_train, corr_test = metrics.correlations(trainPredict,trainY,testPredict,testY)

   	print "Corr_train: " + str(corr_train[0,1]) + " " + "Corr_test: " + str(corr_test[0,1])


   	if corr_train[0,1] >= 0.8 and corr_test[0,1] >= 0.8:
	   	
	  	collapse_layer_2gates(collapse_matrix_2gates,model.layers[1].get_weights()[1],units)
	  	collapse_layer_2gates(collapse_matrix_2gates_mh,model.layers[1].get_weights()[1],units,mean_method='armonic')
	  	collapse_layer_2gates(collapse_matrix_2gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric')

	  	collapse_layer_3gates(collapse_matrix_3gates,model.layers[1].get_weights()[1],units)
	  	collapse_layer_3gates(collapse_matrix_3gates_mh,model.layers[1].get_weights()[1],units,mean_method='armonic')
	  	collapse_layer_3gates(collapse_matrix_3gates_mg,model.layers[1].get_weights()[1],units,mean_method='geometric')

			
		label_recurrent_kernel = case + "_recurrent_kernel"
		label_matrix.append(label_recurrent_kernel)
			
	label_recurrent_kernel = case + "_recurrent_kernel"
	label_matrix.append(label_recurrent_kernel)

   	save.save_model(model,units,layers,case,order)
   	save.save_txt(corr_train,corr_test,units,layers,case,order)
   	save.plot_model_predictions(trainPredict,trainY,testPredict,testY,corr_train,corr_test,units,layers,case,order)














