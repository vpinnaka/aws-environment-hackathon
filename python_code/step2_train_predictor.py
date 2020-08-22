
import numpy as np
from scipy.signal import lfilter
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# targetDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
targetDeviceCPU = torch.device('cpu')
targetDeviceGPU = torch.device('cuda:0') 
targetDevice = targetDeviceCPU


# Split train/val/test data
samplesInADay = 96 # 96 samples 15 minutes apart = 24 hours 

npTrainMatrix = np.load('npTrainMatrix.npy')
n = len(npTrainMatrix)

X_train = npTrainMatrix #[:int(n*0.75), :]
# X_test = npTrainMatrix[int(n*0.75):, :]

import torch, torch.nn as nn, time
from torch.utils.data import Dataset, DataLoader

dataLoaderTrain = DataLoader( X_train.astype('float32'), 
                              batch_size = 128, 
                              shuffle = True ) # could adjust batch size

# dataLoaderTest = DataLoader( X_test.astype('float32'), 
#                              batch_size = 1, 
#                              shuffle = False )


inputDimensionality = X_train.shape[1]

model = nn.Sequential (
    nn.Linear(inputDimensionality, inputDimensionality//2), nn.Sigmoid(),
    nn.Linear(inputDimensionality//2, inputDimensionality//4), nn.Sigmoid(),
    nn.Linear(inputDimensionality//4, inputDimensionality//10), nn.Sigmoid(),
    nn.Linear(inputDimensionality//10, inputDimensionality//4), nn.Sigmoid(),
    nn.Linear(inputDimensionality//4, inputDimensionality//2), nn.Sigmoid(),
    nn.Linear(inputDimensionality//2, inputDimensionality)
)

# USE RELU fn
# model = nn.Sequential (
#     nn.Linear(inputDimensionality, inputDimensionality//2), nn.ReLU(),
#     nn.Linear(inputDimensionality//2, inputDimensionality//4), nn.ReLU(),
#     nn.Linear(inputDimensionality//4, inputDimensionality//10), nn.ReLU(),
#     nn.Linear(inputDimensionality//10, inputDimensionality//4), nn.ReLU(),
#     nn.Linear(inputDimensionality//4, inputDimensionality//2), nn.ReLU(),
#     nn.Linear(inputDimensionality//2, inputDimensionality)
# )

def train_model ( model, dataLoader, targeDevice, nEpochs = 10 ):

    model = model.to( targetDevice )
    
#     lossFunction = nn.MSELoss()
    lossFunction = nn.L1Loss()
    optimizer = torch.optim.Adam( model.parameters())
    lossHistory = []
    history = dict(train=[], val=[])
    batchLoss = []
    
    # training loop    
    for iEpoch in range(nEpochs):   
        cumulativeLoss = 0
        for i, iInputBatch in enumerate( dataLoader ):
            
            # apply filter
            n = 20  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n
            a = 1
            iInputBatch = torch.from_numpy(lfilter(b,a,iInputBatch)).float()

            # move batch data to target training device [ cpu or gpu ]
            iInputBatch = iInputBatch.to( targetDevice )

            # zero/reset the parameter gradient buffers to avoid accumulation [ usually accumulation is necessary for temporally unrolled networks ]
            optimizer.zero_grad()

            # generate predictions/reconstructions
            predictions = model.forward(iInputBatch)

            # compute error 
            loss = lossFunction( predictions, iInputBatch )
            cumulativeLoss += loss.item() # gets scaler value held in the loss tensor
            
            # compute gradients by propagating the error backward through the model/graph
            loss.backward()

            # apply gradients to update model parameters
            optimizer.step()
            batchLoss.append(loss.item())
        print( 'epoch {} of {} -- avg batch loss: {}'.format(iEpoch, nEpochs, cumulativeLoss))
        
        lossHistory += [ cumulativeLoss ]
    return model, lossHistory, batchLoss



model, lossHistory, batchLoss = train_model( model, dataLoaderTrain, targetDevice, nEpochs = 10 )

# print('elapsed time : {} '.format(time.time() - startTime))
# MODEL_PATH = 'model_RELU.pth'
# torch.save(model, MODEL_PATH)

#%% visualize progression of learning
plt.figure(figsize=(6,3))
plt.plot(lossHistory)
plt.title('Loss History'); plt.xlabel('epoch'); plt.ylabel('cumulative loss');
