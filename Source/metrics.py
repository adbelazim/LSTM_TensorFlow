#import keras.backend as K
import numpy as np
from tensorflow.contrib.metrics.python.ops.metric_ops import streaming_pearson_correlation as pearson

def correlations(trainPredict,trainY,testPredict,testY):
    corr_train = np.corrcoef(trainPredict[:,0], trainY[0,:])
    corr_test = np.corrcoef(testPredict[:,0], testY[0,:])
    return corr_train, corr_test





