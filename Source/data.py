import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config as cfg
import config_keras as cfg_keras
from data_utils import lectura, lineal_interpolation, create_timesteps

def train_test_data(reshape_option = 1):
   files_config = cfg.get_files_config()
   nn_config = cfg_keras.get_nn_config()
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
   scaler_cbfv = MinMaxScaler(feature_range=(0, 1))
   cbfv_train = scaler_cbfv.fit_transform(cbfv_train)
   cbfv_test = scaler_cbfv.fit_transform(cbfv_test)
   scaler_abp = MinMaxScaler(feature_range=(0, 1))
   abp_train = scaler_abp.fit_transform(abp_train)
   abp_test = scaler_abp.fit_transform(abp_test)
   

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

   return trainX, testX, trainY, testY, scaler_cbfv, scaler_abp
   

   
   
   







