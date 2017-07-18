import mds_visualization
import run_network
import two_recurrent_layers
from os import listdir
import os
import config_keras as cfg_keras
import config as cfg

import global_queue

def get_subjects(path_subjects):
	path_subjects = path_subjects
	subjects = listdir(path_subjects)
	return subjects

def get_subject_files(path_subjects,subject):
	path_subject_files = path_subjects  + subject
	files = listdir(path_subject_files)
	return files

def get_parameters(files):
	orders = []
	cases = []
	for f in files:
		if f != ".DS_Store":
			# remove .txt extension
			f = f.strip(".txt")
			# first number is the order file
			if f[0] not in orders:
				orders.append(f[0]) 
			# case is the next two numbers
			if (f[1]+f[2]) not in cases:
				cases.append((f[1]+f[2]))
		
	return orders, cases

def create_collapse_final_matrix(collapse_final_matrix, collapse_matrix):
	for vector in collapse_matrix:
		collapse_final_matrix.append(vector)
	return collapse_final_matrix

def save_weights_matrix(weight_matrix,path_save,units,layers,name_file):
	dir_save = path_save + str(units) + "units" + "_" + "layers" +str(layers)
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)

	file_save = dir_save + "/" +name_file
	with open(file_save,"w") as f:
		for vector in weight_matrix:
			for element in vector:
				f.write(str(element))
				f.write(" ")

			f.write("\n")

def save_label_matrix(label_matrix,path_save,units,layers,name_file):
	dir_save = path_save + str(units) + "units" + "_" + "layers" +str(layers)
	if not os.path.exists(dir_save):
		os.makedirs(dir_save)

	file_save = dir_save + "/" +name_file
	with open(file_save,"w") as f:
		for element in label_matrix:
			f.write(str(element))
			f.write("\n")

def process_subjects():



	#Matrixs for plot
	collapse_final_matrix_2gates = []
	collapse_final_matrix_2gates_flat = []
	collapse_final_matrix_2gates_mg = []

	collapse_final_matrix_3gates = []
	collapse_final_matrix_3gates_flat = []
	collapse_final_matrix_3gates_mg = []

	collapse_final_matrix_4gates = []
	collapse_final_matrix_4gates_flat = []
	collapse_final_matrix_4gates_mg = []

	label_final_matrix = []

	grid_config = cfg.get_grid_config()
	path_subjects = grid_config['path_subjects']
	#subjects = get_subjects("../Subjects_long_simulated/")
	subjects = get_subjects(path_subjects)
	i = 0
	print(subjects)
	#for subject in subjects[1:len(subjects)]:
	for subject in subjects:
		if subject != ".DS_Store":
			print("################################################")
			print("SUBJECT " + str(i))
			print("################################################")
			print()
			files = get_subject_files(path_subjects,subject)
			print(files)
			orders, cases = get_parameters(files)
			print(orders)
			print(cases)

			#push subject to queue
			global_queue.push_actual_subject(subject,global_queue.subjects_list)
			collapse_matrix_2gates, collapse_matrix_2gates_flat, collapse_matrix_2gates_mg, collapse_matrix_3gates, collapse_matrix_3gates_flat, collapse_matrix_3gates_mg, collapse_matrix_4gates, collapse_matrix_4gates_flat, collapse_matrix_4gates_mg, label_matrix = two_recurrent_layers.two_recurrent_layers(subject,cases,orders)
			global_queue.pop_actual_subject(global_queue.subjects_list)
			#pop subject from queue


			collapse_final_matrix_2gates = create_collapse_final_matrix(collapse_final_matrix_2gates,collapse_matrix_2gates)
			collapse_final_matrix_2gates_flat = create_collapse_final_matrix(collapse_final_matrix_2gates_flat,collapse_matrix_2gates_flat)
			collapse_final_matrix_2gates_mg = create_collapse_final_matrix(collapse_final_matrix_2gates_mg,collapse_matrix_2gates_mg)
			
			collapse_final_matrix_3gates = create_collapse_final_matrix(collapse_final_matrix_3gates,collapse_matrix_3gates)
			collapse_final_matrix_3gates_flat = create_collapse_final_matrix(collapse_final_matrix_3gates_flat,collapse_matrix_3gates_flat)
			collapse_final_matrix_3gates_mg = create_collapse_final_matrix(collapse_final_matrix_3gates_mg,collapse_matrix_3gates_mg)

			collapse_final_matrix_4gates = create_collapse_final_matrix(collapse_final_matrix_4gates,collapse_matrix_4gates)
			collapse_final_matrix_4gates_flat = create_collapse_final_matrix(collapse_final_matrix_4gates_flat,collapse_matrix_4gates_flat)
			collapse_final_matrix_4gates_mg = create_collapse_final_matrix(collapse_final_matrix_4gates_mg,collapse_matrix_4gates_mg)

			label_final_matrix = create_collapse_final_matrix(label_final_matrix,label_matrix)

			i+=1

	nn_config = cfg_keras.get_nn_config()

	df_1 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_2gates,label_final_matrix)
	df_2 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_2gates_flat,label_final_matrix)

	df_4 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_3gates,label_final_matrix)
	df_5 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_3gates_flat,label_final_matrix)

	df_7 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_4gates,label_final_matrix)
	df_8 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_4gates_flat,label_final_matrix)

	#Se guardan las matrices
	save_weights_matrix(collapse_final_matrix_2gates,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"2gates_aritmetic.txt")
	save_weights_matrix(collapse_final_matrix_2gates_flat,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"2gates_flat.txt")

	save_weights_matrix(collapse_final_matrix_3gates,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"3gates_aritmetic.txt")
	save_weights_matrix(collapse_final_matrix_3gates_flat,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"3gates_flat.txt")

	save_weights_matrix(collapse_final_matrix_4gates,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"4gates_aritmetic.txt")
	save_weights_matrix(collapse_final_matrix_4gates_flat,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"4gates_flat.txt")

	save_label_matrix(label_final_matrix,"../Weight_matrix/",nn_config['units'],nn_config['layers'],"label_matrix.txt")

	mds_visualization.visualization(df_1,"distancia_euclidean_2gates_aritmetic")
	mds_visualization.visualization(df_2,"distancia_euclidean_2gates_flat")

	mds_visualization.visualization(df_4,"distancia_euclidean_3gates_aritmetic")
	mds_visualization.visualization(df_5,"distancia_euclidean_3gates_flat")

	mds_visualization.visualization(df_7,"distancia_euclidean_4gates_aritmetic")
	mds_visualization.visualization(df_8,"distancia_euclidean_4gates_flat")





if __name__ == "__main__":
	process_subjects()























	
