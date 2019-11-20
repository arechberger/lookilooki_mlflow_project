import bz2
import pickle
import random
import math
import numpy as np

def get_data():
    traindata = pickle.load(bz2.BZ2File('data/train.pickle.bz2','r'))
    random.seed(12034123)
    random.shuffle(traindata)
    validation_data = traindata[:math.floor(len(traindata)*0.1)]
    traindata = traindata[math.floor(len(traindata)*0.1):]
    traindata_uz = list(zip(*traindata))
    validation_data_uz = list(zip(*validation_data))
    train_x, train_y = traindata_uz[0], traindata_uz[1]
    val_x, val_y = validation_data_uz[0], validation_data_uz[1]
    
    return np.asarray(train_x,dtype=np.float32)[:,None,:,:],np.asarray(train_y,dtype=np.int64),np.asarray(val_x,dtype=np.float32)[:,None,:,:],np.asarray(val_y,dtype=np.int64)
