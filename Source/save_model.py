from keras.utils import plot_model
import config as cfg
import matplotlib.pyplot as plt


def save_model(model,units,layers):
   files_save = cfg.get_files_save(units,layers)
   path_model = files_save['file_save'] + "/"
   path_model = path_model + "model.png"

   #Plot model in png format
   plot_model(model, to_file=path_model,show_shapes=True)
   model.save_weights(files_save['file_weights'])
   print model.summary()

   #save model in h5 file
   path_h5 = files_save['file_save'] + "/" + 'my_model.h5'
   model.save(path_h5)  

   #Model to json
   json_string = model.to_json() 
   path_json = files_save['file_save'] + "/" +"model.json"
   model_json = open(path_json,'w')
   model_json.write(str(json_string))
   model_json.close()

def save_txt(corr_train,corr_test,units,layers):
      model_correlation = str(corr_train[0,1]) + " " + str(corr_test[0,1])
      model_units_layers = "Layers: " + str(layers) + " "+ "Units: " + str(units)

      files_save = cfg.get_files_save(units,layers)
      path_txt = files_save['file_txt']
      model_txt = open(path_txt,'w')
      model_txt.write(model_correlation)
      model_txt.write("\n")
      model_txt.write(model_units_layers)
      model_txt.close()


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











