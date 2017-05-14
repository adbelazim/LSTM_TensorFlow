import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config as cfg

def lectura(filename):

   file = open(filename, 'r')
   linea = []
   lineas = []
   for line in file:
      linea = line.split('\t')
      i = 0
      for data in linea:
         linea[i] = (data.strip())
         i = i+1
      lineas.append(linea)
   lineas = np.asarray(lineas)  
   return lineas

def lineal_interpolation(vector):
   new_vector = np.zeros((len(vector)/2,1))
   for index in range(0,len(vector)-1,2):
      new_vector[index/2,0] = ((vector[index] + vector[index+1])/2)
   return new_vector


def create_timesteps(dataset, time_steps=1):
   dataX = []
   for i in range(len(dataset)-time_steps-1):
      a = dataset[i:(i+time_steps), 0]
      dataX.append(a)
   return np.array(dataX)


def train_test_data(reshape_option = 1):
   files_config = cfg.get_files_config()
   nn_config = cfg.get_nn_config()
# Lectura de datos de entrenamiento
   data_train = lectura(files_config['filename_train'])
  
   # Lectura de datos de test
   data_test = lectura(files_config['filename_test'])
  
   #transformacion de datos de entrenamiento y test en arreglos de float
   time_train,cbfv_train,abp_train = np.split(data_train,3,axis = 1)
   cbfv_train = np.asarray(cbfv_train, dtype=np.float64)
   abp_train = np.asarray(abp_train, dtype=np.float64)
 
   time_test,cbfv_test,abp_test = np.split(data_test,3,axis = 1)
   cbfv_test = np.asarray(cbfv_test, dtype=np.float64)
   abp_test = np.asarray(abp_test, dtype=np.float64)
   
   #se aplica interpolacion lineal a los datos
   cbfv_train = lineal_interpolation(cbfv_train)
   abp_train = lineal_interpolation(abp_train)

   cbfv_test = lineal_interpolation(cbfv_test)
   abp_test = lineal_interpolation(abp_test)     

   #Se normalizan los datos para facilitar la convergencia del gradiente
   # normalize the dataset
   scaler = MinMaxScaler(feature_range=(0, 1))
   abp_train = scaler.fit_transform(abp_train)
   abp_test = scaler.fit_transform(abp_test)
   cbfv_train = scaler.fit_transform(cbfv_train)
   cbfv_test = scaler.fit_transform(cbfv_test)

   # reshape into X=t and Y=t+1
   trainX = create_timesteps(abp_train, nn_config['time_steps'])
   trainY = cbfv_train[nn_config['time_steps']-1:len(cbfv_train)-2,0]

   testX = create_timesteps(abp_test, nn_config['time_steps'])
   testY = cbfv_test[nn_config['time_steps']-1:len(cbfv_test)-2,0]

   #reshape_option = 1 significa que se utilizaran los retardos como time_steps, de lo contrario los retardos equivalen a la dimension de los datos
   if reshape_option == 1:
      # Se considera look_back como la cantidad de time_steps dejando 1 sola feature
      trainX = np.reshape(trainX, (trainX.shape[0], nn_config['time_steps'], nn_config['input_dim']))
      testX = np.reshape(testX, (testX.shape[0], nn_config['time_steps'], nn_config['input_dim']))
   else:
      trainX = np.reshape(trainX, (trainX.shape[0], nn_config['input_dim'], nn_config['time_steps']))
      testX = np.reshape(testX, (testX.shape[0], nn_config['input_dim'], nn_config['time_steps']))

   return trainX, testX, trainY, testY, scaler
   

   
   
   







