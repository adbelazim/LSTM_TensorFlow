import mds_visualization
import run_network
import two_recurrent_layers
from os import listdir

def get_subjects(path_subjects):
	path_subjects = path_subjects
	subjects = listdir(path_subjects)
	return subjects

def get_subject_files(subject):
	path_subjects = "../Subjects_simulated/"
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

	subjects = get_subjects("../Subjects_simulated/")
	i = 0
	print(subjects)
	#for subject in subjects[1:len(subjects)]:
	for subject in subjects:
		if subject != ".DS_Store":
			print("################################################")
			print("SUBJECT " + str(i))
			print("################################################")
			print()
			files = get_subject_files(subject)
			print(files)
			orders, cases = get_parameters(files)
			print(orders)
			print(cases)
			collapse_matrix_2gates, collapse_matrix_2gates_flat, collapse_matrix_2gates_mg, collapse_matrix_3gates, collapse_matrix_3gates_flat, collapse_matrix_3gates_mg, collapse_matrix_4gates, collapse_matrix_4gates_flat, collapse_matrix_4gates_mg, label_matrix = two_recurrent_layers.two_recurrent_layers(subject,cases,orders)
			
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

	

	df_1 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_2gates,label_final_matrix)
	df_2 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_2gates_flat,label_final_matrix)
	df_3 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_2gates_mg,label_final_matrix)

	df_4 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_3gates,label_final_matrix)
	df_5 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_3gates_flat,label_final_matrix)
	df_6 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_3gates_mg,label_final_matrix)

	df_7 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_4gates,label_final_matrix)
	df_8 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_4gates_flat,label_final_matrix)
	df_9 = mds_visualization.reduction_dimensionality_2(collapse_final_matrix_4gates_mg,label_final_matrix)

	mds_visualization.visualization(df_1,"distancia_euclidean_2gates")
	mds_visualization.visualization(df_2,"distancia_euclidean_2gates_flat")
	mds_visualization.visualization(df_3,"distancia_euclidean_2gates_mg")

	mds_visualization.visualization(df_4,"distancia_euclidean_3gates")
	mds_visualization.visualization(df_5,"distancia_euclidean_3gates_flat")
	mds_visualization.visualization(df_6,"distancia_euclidean_3gates_mg")

	mds_visualization.visualization(df_7,"distancia_euclidean_4gates")
	mds_visualization.visualization(df_8,"distancia_euclidean_4gates_flat")
	mds_visualization.visualization(df_9,"distancia_euclidean_4gates_mg")


if __name__ == "__main__":
	process_subjects()























	
