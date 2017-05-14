from keras.utils import plot_model
import config as cfg
import matplotlib.pyplot as plt

def save_model(model,units,layers):
   	files_save = cfg.get_files_save(units,layers)

	path_model = files_save['file_save'] + "/"
   	path_model = path_model + "model.png"
   	plot_model(model, to_file=path_model,show_shapes=True)

   	model.save_weights(files_save['file_weights'])

   	print model.summary()

   	model.save('my_model.h5')  
   	json_string = model.to_json() 
   	path_json = files_save['file_save'] + "/" +"model.json"
   	model_json = open(path_json,'w')
   	model_json.write(str(json_string))
   	model_json.close()


def plot_model_predictions(trainPredict,trainY,testPredict,testY,corr_train,corr_test,units,layers):
   ##PLOTS del modelo

   	files_save = cfg.get_files_save(units,layers)
   	
   	title1 = repr(corr_train[0,1]) 
   	title1 = 'train ' + title1  
   	title2 = repr(corr_test[0,1])
   	title2 = 'test ' + title2  

   	plt.figure(2)
   	plt.subplot(111)
   	plt.title(title1)
   	plt.plot(trainY[0,:], 'r', trainPredict[:,0], 'b')
   	plt.ylabel("CBFV")
   	plt.xlabel("Time")
   	plt.savefig(files_save['file_save'] + "/" +'trainPredict.png')

   	plt.figure(3)
   	plt.subplot(111)
   	plt.title(title2)
   	plt.plot(testY[0,:], 'r', testPredict[:,0], 'b')
   	plt.ylabel("CBFV")
   	plt.xlabel("Time")
   	plt.savefig(files_save['file_save'] + "/" +'testPredict.png')  











