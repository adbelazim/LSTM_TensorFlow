import numpy as np

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

def fold_order(trainX, testX, trainY, testY,order,nn_config):

   if order == "1":
      print("entre a order 1")
      trainX = np.reshape(trainX, (trainX.shape[0], nn_config['time_steps'], nn_config['input_dim']))
      testX = np.reshape(testX, (testX.shape[0], nn_config['time_steps'], nn_config['input_dim']))
      return trainX, testX, trainY, testY
   else:
      print("entre a order 2")
      trainX = np.reshape(trainX, (trainX.shape[0], nn_config['time_steps'], nn_config['input_dim']))
      testX = np.reshape(testX, (testX.shape[0], nn_config['time_steps'], nn_config['input_dim']))
      return testX, trainX, testY, trainY


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












