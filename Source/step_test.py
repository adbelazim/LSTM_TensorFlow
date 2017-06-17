import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from data import create_timesteps
import config as cfg
import config_keras as cfg_keras

def generate_step(sampling_time = 0.2, 
                left_stabilisation_time = 30,
                time_until_release = 10,
                time_after_release = 20,
                smoth_step_stimulus = False,
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

    print(int(samples_stabilisation))
    print(samples_after_release)
    print(samples_until_release)
    print(samples_left)

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

def plot_step_test(prediction,units,layers,name):
    files_save = cfg.get_files_save(units,layers)
    plt.figure(1)
    plt.title("Step test")
    plt.plot(prediction, 'b')
    plt.ylabel("CBFV")
    plt.xlabel("Time")
    plt.savefig(files_save['file_save'] + "/" +name)  

def step_test(model,step):
    nn_config = cfg_keras.get_nn_config()
    step = np.reshape(step, (len(step),1))
    step = create_timesteps(step,nn_config['time_steps'])
    step = np.reshape(step,(step.shape[0],nn_config['time_steps'], nn_config['input_dim']))
    prediction = model.predict(step)
    return prediction

