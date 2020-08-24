import numpy as np
import torch
import matplotlib.pylab as plt
import joblib
import pandas as pd
from scipy.signal import lfilter


# MODEL_PATH = 'model.pth'
MODEL_PATH = 'model_RELU_v5.pth'
model = torch.load(MODEL_PATH)


n = 20  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

test_data = joblib.load('../test_data/test_dataset_2019.numpy')
test_data2 = lfilter(b,a,test_data)
testingData = test_data2.astype(np.float32)

x_test_pred = model.forward(torch.from_numpy(testingData))

# test_mae_loss = np.median(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), axis = 1)
test_mae_loss = np.quantile(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), 0.95,axis = 1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.figure(figsize=(6,3)) 
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

#%%
idx = (-test_mae_loss).argsort() # predicted windows where anomaly exists
pred = np.zeros(len(testingData))
pred[idx[:200]] = 1

# examine anomalies in the test data set
for i in range(1):
  plt.figure(figsize=(10, 3))
  # plt.plot(test_data[idx[i], :])
  plt.plot(testingData[idx[i], :],color='r')
  plt.plot(x_test_pred.detach().numpy()[idx[i],:])
  
#%% find where anomaly starts  

error = np.abs(x_test_pred.detach().numpy()[idx[:200], :16*96] - testingData[idx[:200], :16*96])
errorWhen = np.zeros(200)
errorPeriod = []


for i in range(200):
    threshold = np.quantile(error[i], 0.95)
    errorPeriod.append(np.where((error[i] >= threshold).tolist()))
    # consecutive 20+ points anomalous
    # maxErrorLoc = np.where(error[i] == np.max(error[i]))[0][0]   
    # while (maxErrorLoc in np.array(errorPeriod[i])):
    #     maxErrorLoc -= 1
    err = np.array(errorPeriod[i])
    
    maxErrorLoc = np.median(errorPeriod[i])
    
    for j in np.array(sorted(err)).T:
        # isAnomalous = sorted(err) ==  np.array(range(np.min(err), np.max((err))+1))
        if (j+1) in np.array(sorted(err)) and (j+2) in np.array(sorted(err)) and \
            (j+3) in np.array(sorted(err)) and (j+4) in np.array(sorted(err)) and \
                 (j+5) in np.array(sorted(err)) and (j+6) in np.array(sorted(err))\
            and (j+7) in np.array(sorted(err)) and (j+8) in np.array(sorted(err))\
            and (j+9) in np.array(sorted(err)) and (j+10) in np.array(sorted(err)) and\
                (j+11) in np.array(sorted(err)) and (j+12) in np.array(sorted(err)):
            maxErrorLoc = j
            break
    
    errorWhen[i] = maxErrorLoc

#%% visualize anomalies
ii = 1

for ii in range(5):
    plt.figure(figsize=(10,3))
    plt.plot(test_data[idx[ii], :], linewidth = 3, label ='raw')
    plt.plot(x_test_pred.detach().numpy()[idx[ii],:], linewidth = 3, label = 'predict')
    
    if ii < 200:
        plt.plot( np.array((errorPeriod[ii])).T, test_data[idx[ii],errorPeriod[ii]].T,'.',color='y',\
                 linewidth = 3, label = 'anomaly')
        
        plt.plot( errorWhen[ii], test_data[idx[ii],int(errorWhen[ii])],'o',markersize=10, markerfacecolor='r',
             markeredgewidth=.5, markeredgecolor='k', label = 'start')
    plt.legend()
    
#%%
from matplotlib import pyplot as plt
from celluloid import Camera
import matplotlib.animation as animation
import imageio

# fig = plt.figure(figsize=(10,3))
# camera = Camera(fig)
# for ii in range(200):
#     plt.plot(test_data[idx[ii], :], linewidth = 3, label ='raw', color = 'b')
#     plt.plot(x_test_pred.detach().numpy()[idx[ii],:], linewidth = 3, label = 'predict', color = 'r')
#     plt.plot( np.array((errorPeriod[ii])).T, test_data[idx[ii],errorPeriod[ii]].T,'.',color='y',\
#                  linewidth = 3, label = 'anomaly')
        
#     plt.plot( errorWhen[ii], test_data[idx[ii],int(errorWhen[ii])],'o',markersize=10, markerfacecolor='r',
#          markeredgewidth=.5, markeredgecolor='k', label = 'start')
#     plt.ylim([-4.5, 5])
#     # plt.plot([i] * 10)
#     camera.snap()
# animation = camera.animate()
# animation.save('dynamic_images2.gif')




#%% reconstruction errors
# error = np.abs(x_test_pred.detach().numpy()[idx[:200], :] - testingData[idx[:200], :])
for ii in range(6,7):
    plt.figure(figsize=(10,3))
    plt.plot(error[ii, :], linewidth = 3, label ='error')
    
    
    if ii < 200:
        plt.plot( np.array((errorPeriod[ii])).T, error[ii,errorPeriod[ii]].T,'.',color='y',\
                 linewidth = 3, label = 'anomaly')
        
        plt.plot( errorWhen[ii], error[ii,int(errorWhen[ii])],'o',markersize=10, markerfacecolor='r',
             markeredgewidth=.5, markeredgecolor='k', label = 'start')
    plt.legend()
    
#%% Generate output
df = pd.DataFrame(np.array([1+idx[:200].astype(int), \
                            errorWhen. astype(int)]).T, columns = ['day','time'])
df.to_csv('submission_quantile.txt', index = False)
df.head()

#%%
def parse_submission ( filename ):
    dayIndicators = [] 
    startSampleIndicators = []
    lineCount = 0 
    with open(filename) as submission_text:
        line = submission_text.readline()        
        while line:
            if lineCount > 200: break
                    
            dayIndicators += [line.split(',')[0].strip()]
            startSampleIndicators += [line.split(',')[1].strip()]
            
            print( f'parsing line#{lineCount}: {line}' )
            line = submission_text.readline()
            lineCount += 1
            
    return dayIndicators, startSampleIndicators

# dayIndicators, startSampleIndicators = parse_submission ( 'submission.txt')

#%% compare 3 methods
df_mean = pd.read_csv("submission_mean.txt")
df_median = pd.read_csv("submission_median.txt")
df_quantile = pd.read_csv("submission_quantile.txt")

idx1 = np.array(df_mean["day"])
idx2 = np.array(df_median["day"])
idx3 = np.array(df_quantile["day"])

# idx2_in_idx1 = []
all_in_one = []

for i in idx1:
    if i in idx2 and i in idx3:
        all_in

