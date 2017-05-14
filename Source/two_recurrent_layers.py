import numpy as np
import sys

import run_network

#from sklearn.metrics import mean_squared_error
#from keras.utils import plot_model
#from sklearn.metrics import matthews_corrcoef
#from keras.layers import Input, Dropout
#from keras import metrics

from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
   case = str(sys.argv[1])
   order = str(sys.argv[2])
   layer = int(sys.argv[3])
   unit = int(sys.argv[4])

   print "Case: "+ case + ", " + "Fold: " + order + ", " + "Layers: " + str(layer) + ", " + "Units: " + str(unit)
   print

   run_network.run_network(unit,layer)
   sys.exit()
   #case = str(sys.argv[1])
   #order = str(sys.argv[2])

   