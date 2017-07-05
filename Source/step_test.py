import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from data_utils import lectura, lineal_interpolation, create_timesteps
import config as cfg
import config_keras as cfg_keras

def generate_step(sampling_time = 0.2, 
                left_stabilisation_time = 30,
                time_until_release = 10,
                time_after_release = 20,
                smoth_step_stimulus = True,
                filter_order = 2,
                cutoff_frecuency = 0.2,
                ):
    
    frecuency = 1 / sampling_time

    samples_stabilisation = 0
    if(smoth_step_stimulus):
        samples_stabilisation = left_stabilisation_time / sampling_time

    samples_until_release = time_until_release / sampling_time
    samples_after_release = time_after_release / sampling_time
    samples_left = samples_stabilisation + samples_until_release

    samples_total = samples_left + samples_after_release

    #unos antes de la caida
    left_escalon = np.arange(samples_left)
    left_escalon = np.ones_like(left_escalon)
    #ceros despues de la caida
    right_escalon = np.arange(samples_after_release)
    right_escalon = np.zeros_like(right_escalon)

    escalon = np.concatenate((left_escalon, right_escalon),axis = 0)

    if(smoth_step_stimulus):
        print("entre")
        wn = cutoff_frecuency / (frecuency / 2)
        b, a = signal.butter(filter_order, wn)
        escalon = signal.lfilter(b, a,escalon)

    if(samples_stabilisation > 0):
        escalon = escalon[int(samples_stabilisation):int(samples_total)-1]

    return np.asarray(escalon)

def plot_step_test(prediction,abp,subject,units,layers,case,order,name):
    nn_config = cfg_keras.get_nn_config()
    files_save = cfg.get_files_save(subject,units,layers,case,order,nn_config)
    #se guarda Abp con retardos y prediction en un archivo
    files_save['file_step_txt']

    print(len(abp))
    print(len(prediction))

    with open(files_save['file_step_txt'],"w") as f:
        for i in range(0,len(prediction)):
            f.write(str(abp[i]))
            f.write("\t")
            f.write(str(prediction[i,0]))
            f.write("\n")

    #se crea ploteo
    plt.figure(1)
    plt.title("Step test")
    plt.plot(prediction, 'b')
    plt.ylabel("CBFV")
    plt.xlabel("Time")
    plt.savefig(files_save['file_save'] + "/" + name)  

def step_test(model,step):
    nn_config = cfg_keras.get_nn_config()
    step = np.reshape(step, (len(step),1))
    step = create_timesteps(step,nn_config['time_steps'])
    step = np.reshape(step,(step.shape[0],nn_config['time_steps'], nn_config['input_dim']))
    prediction = model.predict(step)
    return prediction

def step_prediction(abp_first_lag,sampling_time,model,subject,units,layers,case,order,name_plot):

    #get len of abp signal
    files_config = cfg.get_files_config(subject,case)
    data_train = lectura(files_config['filename_train'])

    time_train,cbfv_train,abp_train = np.split(data_train,3,axis = 1)
    time_train = np.asarray(time_train, dtype=np.float64)
    abp_train = np.asarray(abp_train, dtype=np.float64)

    if(sampling_time == 0.2):
        abp_train = lineal_interpolation(abp_train)
        #si se interpolo la frecuencia de muestreo aumenta al doble
        sampling_time = sampling_time*2

    #Values for step test
    sampling_time = sampling_time
    time = len(abp_train)*sampling_time

    #30% of the signal before escalon
    time_after = time*0.7
    time_until = time - time_after

    print("step sampling_time", sampling_time)

    step = generate_step(sampling_time = sampling_time, 
                            time_until_release=time_until ,
                            time_after_release = time_after, 
                            smoth_step_stimulus = True)

    print("len step",len(step))

    step_prediction = step_test(model,step)
    plot_step_test(step_prediction,abp_first_lag,subject,units,layers,case,order,name_plot)

    #step_prediction = scaler_cbfv.inverse_transform(step_prediction)
    #step_test.plot_step_test(step_prediction,trainX[:,0,0],subject,units,layers,case,order,'step_test.png')




















