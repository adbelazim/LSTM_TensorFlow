import numpy as np
import sys

import run_network
import config_keras as cfg_keras

import global_queue

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
      global_queue.push_actual_case(case,global_queue.cases_list)
      #push case to queu
      for order in orders:
         print("order two recurrent", order)
         global_queue.push_actual_order(order,global_queue.orders_list)
         #push order to queue
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
         #pop order from queue
         global_queue.pop_actual_order(global_queue.orders_list)
      global_queue.pop_actual_case(global_queue.cases_list)
      #pop case from queue
   return collapse_matrix_2gates, collapse_matrix_2gates_flat, collapse_matrix_2gates_mg, collapse_matrix_3gates, collapse_matrix_3gates_flat, collapse_matrix_3gates_mg,collapse_matrix_4gates, collapse_matrix_4gates_flat, collapse_matrix_4gates_mg, label_matrix











   
