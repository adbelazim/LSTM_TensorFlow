import numpy as np
import sys

#import mds_visualization
import run_network
import config_keras as cfg_keras
#from sklearn.preprocessing import MinMaxScaler

#if __name__ == "__main__":
def two_recurrent_layers(subject,cases,orders):
   #cases = ["11","19","55","91","99"]
   #orders = ["1","2"]
   nn_config = cfg_keras.get_nn_config()

   layers = [nn_config['layers']]
   units = [nn_config['units']]

   repeat_experiment = 10
   collapse_matrix_2gates = []
   collapse_matrix_2gates_flat = []
   collapse_matrix_2gates_mg = []

   collapse_matrix_3gates = []
   collapse_matrix_3gates_flat = []
   collapse_matrix_3gates_mg = []

   collapse_matrix_4gates = []
   collapse_matrix_4gates_flat = []
   collapse_matrix_4gates_mg = []

   label_matrix = []

   np.random.seed(997)

   for case in cases:
      for order in orders:
         for layer in layers:
            for unit in units:
               print
               print
               print "Case: "+ case + ", " + "Fold: " + order + ", " + "Layers: " + str(layer) + ", " + "Units: " + str(unit)
               print
               run_network.run_network(subject,unit,layer,case,order,collapse_matrix_2gates,
                                                               collapse_matrix_2gates_flat,
                                                               collapse_matrix_2gates_mg,
                                                               collapse_matrix_3gates,
                                                               collapse_matrix_3gates_flat,
                                                               collapse_matrix_3gates_mg,
                                                               collapse_matrix_4gates,
                                                               collapse_matrix_4gates_flat,
                                                               collapse_matrix_4gates_mg,
                                                               label_matrix,stateful = False)
               #for i in range(1,repeat_experiment):
               #   print("experiment: ",i)
               #   run_network.run_network(unit,layer,case,order,stateful = False)

   return collapse_matrix_2gates, collapse_matrix_2gates_flat, collapse_matrix_2gates_mg, collapse_matrix_3gates, collapse_matrix_3gates_flat, collapse_matrix_3gates_mg,collapse_matrix_4gates, collapse_matrix_4gates_flat, collapse_matrix_4gates_mg, label_matrix

   #df_1 = mds_visualization.reduction_dimensionality_2(collapse_matrix_2gates,label_matrix)
   #df_2 = mds_visualization.reduction_dimensionality_2(collapse_matrix_2gates_flat,label_matrix)
   #df_3 = mds_visualization.reduction_dimensionality_2(collapse_matrix_2gates_mg,label_matrix)

   #df_4 = mds_visualization.reduction_dimensionality_2(collapse_matrix_3gates,label_matrix)
   #df_5 = mds_visualization.reduction_dimensionality_2(collapse_matrix_3gates_flat,label_matrix)
   #df_6 = mds_visualization.reduction_dimensionality_2(collapse_matrix_2gates_mg,label_matrix)


   #mds_visualization.visualization(df_1,"distancia_euclidean_2gates")
   #mds_visualization.visualization(df_2,"distancia_euclidean_2gates_flat")
   #mds_visualization.visualization(df_3,"distancia_euclidean_2gates_mg")

   #mds_visualization.visualization(df_4,"distancia_euclidean_3gates")
   #mds_visualization.visualization(df_5,"distancia_euclidean_3gates_flat")
   #mds_visualization.visualization(df_6,"distancia_euclidean_3gates_mg")
               









   
