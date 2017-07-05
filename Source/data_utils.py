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

def get_files_config(case):
   file_data = "../Data/"
   filename_train = ""
   
   filename_train = file_data + "1" + case + ".txt"
   filename_test = file_data + "2" + case + ".txt"

   files_config = {'filename_train' : filename_train,
               'filename_test' : filename_test
               }
   return files_config











