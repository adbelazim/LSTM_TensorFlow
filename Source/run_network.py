import data as dt
import model_fit
import model_lstm as m_lstm
import metrics
import save_model as save
import config as cfg
from keras.utils import plot_model


#TO DO
#import model_lstm_cnn as m_lstm_cnn
#import model_cnn as m_cnn
#import model_distributed as m_dist

def run_network(units, layers, stateful = False):

	#Se cargan los datos de entrenamiento y test
	trainX, testX, trainY, testY, scaler = dt.train_test_data(reshape_option = 1)
	#Se compila el modelo
	model = m_lstm.init_model_lstm(units,layers) 
	#Se entrena el modelo
	model_fit.model_fit(model,trainX, trainY, testX, testY,units,layers,stateful = stateful)

	if stateful == False:
		#Se predice como se ajustan los conjuntos de entrenamiento y test al modelo
		trainPredict = model.predict(trainX)
   		testPredict = model.predict(testX)

	else:
		fit_config = cfg.get_fit_config()
		# make predictions with batch_size
   		trainPredict = model.predict(trainX,batch_size=fit_config['batch_size'])
   		testPredict = model.predict(testX,batch_size=fit_config['batch_size'])

	#Se obtiene el valor no normalizado de las predicciones. Ivert predictions
   	trainPredict = scaler.inverse_transform(trainPredict)
   	trainY = scaler.inverse_transform([trainY])
   	testPredict = scaler.inverse_transform(testPredict)
   	testY = scaler.inverse_transform([testY])

   	#Se calcula la correlacion
   	corr_train, corr_test = metrics.correlations(trainPredict,trainY,testPredict,testY)

   	print "Corr_train: " + str(corr_train[0,1]) + " " + "Corr_test: " + str(corr_test[0,1])

   	save.save_model(model,units,layers)

   	save.plot_model_predictions(trainPredict,trainY,testPredict,testY,corr_train,corr_test,units,layers)













