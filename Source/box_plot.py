from data_utils import lectura, lineal_interpolation
import numpy as np
import matplotlib.pyplot as plt

def data_to_plot(file_train, file_test):
	data_train = lectura(file_train)
	data_test = lectura(file_test)

	time_train,cbfv_train,abp_train = np.split(data_train,3,axis = 1)
	cbfv_train = np.asarray(cbfv_train, dtype=np.float64)
	abp_train = np.asarray(abp_train, dtype=np.float64)

	time_test,cbfv_test,abp_test = np.split(data_test,3,axis = 1)
	cbfv_test = np.asarray(cbfv_test, dtype=np.float64)
	abp_test = np.asarray(abp_test, dtype=np.float64)

	#se aplica interpolacion lineal a los datos
	#cbfv_train = lineal_interpolation(cbfv_train)
	#abp_train = lineal_interpolation(abp_train)

	#cbfv_test = lineal_interpolation(cbfv_test)
	#abp_test = lineal_interpolation(abp_test)  

	while not (len(cbfv_train) % 9) == 0:
		cbfv_train = cbfv_train[0:len(cbfv_train)-1]

	while not (len(cbfv_test) % 9) == 0:
		cbfv_test = cbfv_test[0:len(cbfv_test)-1]

	cbfv_train_aris = np.split(cbfv_train,9, axis = 0)
	cbfv_test_aris = np.split(cbfv_test,9, axis = 0)

	return cbfv_train_aris, cbfv_test_aris


def getBoxPlots(data_plot, case, figure_number, specific=False, specific_index = 0):
	fig = plt.figure(figure_number, figsize=(9,6))

	ax = fig.add_subplot(111)
	##fill color
	bp = ax.boxplot(data_plot, patch_artist = True, showmeans= True,meanline = True)
	titulo = 'Case: ' + case
	fig.suptitle(titulo, fontsize = 20)
	plt.xlabel('ARI', fontsize = 16)
	plt.ylabel('CBFV', fontsize = 16)

	if specific:
		label = 'ARI' + str(specific_index)
		ax.set_xticklabels([label, label, label, label, label, label, label, label, label])
	else:
		ax.set_xticklabels(['ARI 1', 'ARI 2','ARI 3','ARI 4','ARI 5','ARI 6','ARI 7','ARI 8','ARI 9'])
	##cambio el color
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	for box in bp['boxes']:
		box.set(color = '#7570b3', linewidth=2)
		box.set(facecolor = '#1b9e77')
		#cambio	color y ancho de linea para "whiskers"
	for whisker in bp['whiskers']:
		whisker.set(color = '#7570b3', linewidth = 2)
	for cap in bp['caps']:
		cap.set(color = '#7570b3', linewidth= 2)
	for median in bp['medians']:
		median.set(color = '#b2df8a', linewidth= 2)	
	for flier in bp['fliers']:
		flier.set(marker = 'o', color = '#e7298a', alpha = 0.5)

	name = '../Stationary/' + "box_plot_"+ case + '.png'
	fig.savefig(name, bbox_inches = 'tight')


def error_bar(data,case,figure_number,specific=False,specific_index=0):
	fig = plt.figure(figure_number, figsize=(9,6))
	ax = fig.add_subplot(111)
	x_axis = np.arange(1,10,1)
	#calculando meadias y desviaciones para plotear
	means_plot = []
	std_plot = []
	for arr in data:
		means_plot.append(arr.mean())
		std_plot.append(arr.std())

	ax.errorbar(x_axis, means_plot, xerr = 0.3, yerr=std_plot)

	fig.suptitle(case, fontsize = 20)
	plt.xlabel('ARI', fontsize = 16)
	plt.ylabel('CBFV', fontsize = 16)
	
	if specific:
		label = 'ARI' + str(specific_index)
		ax.set_xticklabels([label, label, label, label, label, label, label, label, label])
	else:
		ax.set_xticklabels(['ARI 1', 'ARI 2','ARI 3','ARI 4','ARI 5','ARI 6','ARI 7','ARI 8','ARI 9'])
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	
	name = '../Stationary/' + "mean_plot_" +case + '.png'
	fig.savefig(name, bbox_inches = 'tight')




if __name__=="__main__":

	case_train = "199"
	case_test = "299"

	print("Ploteando caso " + case_train + " y " + case_test)

	file_train = "../Data/" + case_train + ".txt"
	file_test = "../Data/" + case_test + ".txt"

	data_train_plot, data_test_plot = data_to_plot(file_train,file_test)
	
	if case_train == "155" or case_train == "199" or case_train == "111":
		specific = True
		if case_train == "155": specific_index=5
		if case_train == "111": specific_index=1
		if case_train == "199": specific_index=9

		getBoxPlots(data_train_plot, case_train,1,specific, specific_index)
		getBoxPlots(data_test_plot, case_test,2,specific, specific_index) 

		error_bar(data_train_plot, case_train,3,specific, specific_index)
		error_bar(data_test_plot, case_test,4,specific, specific_index)
	else:
		getBoxPlots(data_train_plot, case_train,1)
		getBoxPlots(data_test_plot, case_test,2) 

		error_bar(data_train_plot, case_train,3)
		error_bar(data_test_plot, case_test,4)




































