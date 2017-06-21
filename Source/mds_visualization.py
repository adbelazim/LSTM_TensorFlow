import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt

import scipy.stats


def visualization(df,name_image):
	groups = df.groupby('label')
	#cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
	colors = ['#B0171F','#FF00FF','#0000FF','#C6E2FF','#00FA9A','#FFFF00','#FFA500','#030303','#FCFCFC','#FFBBFF']
	cluster_colors = {}
	i = 0

	for name in df['label']:
		if name == "11_recurrent_kernel":
			cluster_colors[name] = colors[0]
		elif name == "19_recurrent_kernel":
			cluster_colors[name] = colors[1]
		elif name == "55_recurrent_kernel":
			cluster_colors[name] = colors[2]
		elif name == "91_recurrent_kernel":
			cluster_colors[name] = colors[3]
		else:
			cluster_colors[name] = colors[4]

	cluster_names = {}
	for name in df['label']:
		cluster_names[name] = name

	#cluster_names = {0: '19',1: '55',2: '91'}
	fig, ax = plt.subplots(figsize=(25, 15)) # set size
	ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
	for name, group in groups:
		ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
		ax.set_aspect('auto')
		#ax.tick_params(\
		#	axis= 'x',          # changes apply to the x-axis
		#	which='both',      # both major and minor ticks are affected
		#	bottom='off',      # ticks along the bottom edge are off
		#	top='off',         # ticks along the top edge are off
		#	labelbottom='off')
		#ax.tick_params(\
		#	axis= 'y',         # changes apply to the y-axis
		#	which='both',      # both major and minor ticks are affected
		#	left='off',      # ticks along the bottom edge are off
		#	top='off',         # ticks along the top edge are off
		#		labelleft='off')
	ax.legend(numpoints=1)  #show legend with only 1 point
	for i in range(len(df)):
		ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['label'], size=8)  
	fig_name = name_image + ".png"
	ax.set_xticks(np.arange(-1, 1, 0.1))
	ax.set_yticks(np.arange(-1, 1, 0.1))
	plt.grid(True)
	plt.savefig(fig_name)#show the plot

def reduction_dimensionality(collapse_matrix,label_matrix):
	dist = 1 - cosine_similarity(collapse_matrix)
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=997)
	pos = mds.fit_transform(dist)
	xs, ys = pos[:, 0], pos[:, 1]
	df = pd.DataFrame(dict(x=xs, y=ys, label=label_matrix)) 
	return df

def reduction_dimensionality_2(collapse_matrix,label_matrix):
	dist = 1 - euclidean_distances(collapse_matrix)
	mds = MDS(n_components=2, dissimilarity="precomputed", random_state=997)
	pos = mds.fit_transform(dist)
	xs, ys = pos[:, 0], pos[:, 1]
	df = pd.DataFrame(dict(x=xs, y=ys, label=label_matrix)) 
	return df

def mean_list(list_gates):
	mean = sum(list_gates) / float(len(list_gates))
	return mean

def flatten_list(matrix_gates):
	array_gates = np.asarray(matrix_gates)
	return array_gates.flatten().tolist()

def mean_geometric_list(list_gates):
	normalized = (list_gates-min(list_gates))/(max(list_gates)-min(list_gates))
	mean = scipy.stats.mstats.gmean(normalized)
	return mean


def collapse_layer_2gates(collapse_matrix,weight_matrix,units,mean_method='aritmetic'):
	print("iniciando colapse")
	i_gates = []
	f_gates = []
	c_gates = []
	o_gates = []
	for vector in weight_matrix:
		i = 0
		for element in vector:
			if i < units:
				i_gates.append(element)
			elif i < units*2:
				f_gates.append(element)
			elif i < units*3:
				c_gates.append(element)
			else:
				o_gates.append(element)
			i+=1


	if mean_method == 'flatten':
		collapse_matrix.append(flatten_list([f_gates,c_gates]))
	elif mean_method == 'geometric':
		collapse_matrix.append([mean_geometric_list(f_gates),mean_geometric_list(c_gates)])
	else:
		collapse_matrix.append([mean_list(f_gates),mean_list(c_gates)])

	#collapse_matrix.append([mean_list(i_gates),mean_list(f_gates),mean_list(c_gates),mean_list(o_gates)])
	#collapse_matrix.append([mean_list(f_gates),mean_list(c_gates)])

	return collapse_matrix

def collapse_layer_3gates(collapse_matrix,weight_matrix,units,mean_method='aritmetic'):
	print("iniciando colapse")
	i_gates = []
	f_gates = []
	c_gates = []
	o_gates = []
	for vector in weight_matrix:
		i = 0
		for element in vector:
			if i < units:
				i_gates.append(element)
			elif i < units*2:
				f_gates.append(element)
			elif i < units*3:
				c_gates.append(element)
			else:
				o_gates.append(element)
			i+=1

	if mean_method == 'flatten':
		collapse_matrix.append(flatten_list([f_gates,c_gates,o_gates]))
	elif mean_method == 'geometric':
		collapse_matrix.append([mean_geometric_list(f_gates),mean_geometric_list(c_gates),mean_geometric_list(o_gates)])
	else:
		collapse_matrix.append([mean_list(f_gates),mean_list(c_gates),mean_list(o_gates)])
	#collapse_matrix.append([mean_list(i_gates),mean_list(f_gates),mean_list(c_gates),mean_list(o_gates)])
	#collapse_matrix.append([mean_list(f_gates),mean_list(c_gates),mean_list(o_gates)])

	return collapse_matrix




