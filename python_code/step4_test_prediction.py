import numpy as np
import torch
import matplotlib.pylab as plt
import joblib
import pandas as pd
from scipy.signal import lfilter


# MODEL_PATH = 'model.pth'
MODEL_PATH = 'models/model_RELU_v7.pth'
model = torch.load(MODEL_PATH)


n = 20  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

test_data = joblib.load('../test_data/test_dataset_2019.numpy')
test_data2 = lfilter(b,a,test_data)
testingData = test_data2.astype(np.float32)

x_test_pred = model.forward(torch.from_numpy(testingData))

# test_mae_loss = np.median(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), axis = 1)
test_mae_loss = np.quantile(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), 0.9,axis = 1)
# test_mae_loss = np.mean(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy())**2, axis = 1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.figure(figsize=(6,3)) 
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

#%%
idx = (-test_mae_loss).argsort() # predicted windows where anomaly exists
# idx_in_subset = (-test_mae_loss[np.array(all_in_one)-1]).argsort()[:200] 
# idx = np.array(all_in_one)[idx_in_subset] - 1

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
    threshold = np.quantile(error[i], 0.9)
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

for ii in range(220,225):
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

fig = plt.figure(figsize=(10,3))
camera = Camera(fig)
for ii in range(300,400):
    plt.plot(test_data[idx[ii], :], linewidth = 3, label ='raw', color = 'b')
    plt.plot(x_test_pred.detach().numpy()[idx[ii],:], linewidth = 3, label = 'predict', color = 'r')
    # plt.plot( np.array((errorPeriod[ii])).T, test_data[idx[ii],errorPeriod[ii]].T,'.',color='y',\
    #               linewidth = 3, label = 'anomaly')
        
    # plt.plot( errorWhen[ii], test_data[idx[ii],int(errorWhen[ii])],'o',markersize=10, markerfacecolor='r',
    #       markeredgewidth=.5, markeredgecolor='k', label = 'start')
    plt.ylim([-4.5, 5])
    # plt.plot([i] * 10)
    camera.snap()
animation = camera.animate()
animation.save('dynamic_images2.gif')




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
                            1 + errorWhen[:200]. astype(int)]).T, columns = ['day','time'])
df.to_csv('submission_v7_nofilter.txt', index = False)
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
# !cd /experiments
# df = pd.read_csv("submission.txt")
df1 = pd.read_csv("submission_filter_mean.txt")
df2 = pd.read_csv("submission_nofilter_mean.txt")
df3 = pd.read_csv("submission_filter_relu.txt")
df4 = pd.read_csv("submission_nofilter_relu.txt")
df5 = pd.read_csv("submission_filter_model.txt")
df6 = pd.read_csv("submission_nofilter_model.txt")
df7 = pd.read_csv("submission_filter_v4.txt")
df8 = pd.read_csv("submission_nofilter_v4.txt") # quantile loss
df9 = pd.read_csv("submission_filter_median.txt")
df10 = pd.read_csv("submission_nofilter_median.txt") # median loss
df11 = pd.read_csv("submission_filter_v3.txt")
df12 = pd.read_csv("submission_nofilter_v3.txt") # quantile loss
df13 = pd.read_csv("submission_v6_nofilter.txt") # larger training set w/ test
df14 = pd.read_csv("submission_v6_filter.txt")

idx0 = np.array(df["day"])
idx1 = np.array(df1["day"])
idx2 = np.array(df2["day"])
idx3 = np.array(df3["day"])
idx4 = np.array(df4["day"])
idx5 = np.array(df5["day"])
idx6 = np.array(df6["day"])
idx7 = np.array(df7["day"])
idx8 = np.array(df8["day"])
idx9 = np.array(df9["day"])
idx10 = np.array(df10["day"])
idx11 = np.array(df11["day"])
idx12 = np.array(df12["day"])
idx13 = np.array(df13["day"])
idx14 = np.array(df14["day"])


all_in_one = []
# for i in range(len(test_data)+1):
#     if i in idx3[:150] or (i in idx3[:250] and i in idx4[:250]) or (i in idx7[:250] and i in idx8[:250]) or\
#         (i in idx11[:60] and i in idx12[:60]) or (i in idx9[:60] and i in idx10[:60]) \
#             or (i in idx1[:50] and i in idx2[:50]):
#         # if ((i in idx3 and i in idx4) and (i in idx1 or i in idx5 or i in idx7)):# or (i in idx5 and i in idx6) or (i in idx1 and i in idx2):
#           # if((i in idx1 and i in idx2) or (i in idx3 and i in idx4) or (i in idx5 and i in idx6) or (i in idx7 and i in idx8)): 
#             all_in_one.append(i)


# for i in range(len(test_data)+1):
#     cnt = 0
#     if i in idx3[:150]:
#         cnt += 2
#     if (i in idx3[:250] and i in idx4[:250]): 
#           cnt += 2
#     if (i in idx7[:250] and i in idx8[:250]):
#         cnt += 2
#     if (i in idx11[:60] and i in idx12[:60]):
#           cnt += 2
#     if (i in idx9[:60] and i in idx10[:60]):
#           cnt += 2
#     if (i in idx1[:50] and i in idx2[:50]):
#           cnt += 2
#     if cnt > 1:
#         all_in_one.append(i)

for i in range(len(test_data)+1):    
    if (i in idx3 and i in idx4) or (i in idx13 and i in idx14) or (i in idx7[:200] and i in idx8[:200]):
        all_in_one.append(i)

     
#%%
from matplotlib import pyplot as plt
from celluloid import Camera
import matplotlib.animation as animation
import imageio

fig = plt.figure(figsize=(10,3))
camera = Camera(fig)

for ii in range(len(all_in_one)):
    plt.plot(test_data[all_in_one[ii]-1, :], linewidth = 3, label ='raw', color = 'b')
    plt.plot(x_test_pred.detach().numpy()[all_in_one[ii]-1,:], linewidth = 3, label = 'predict', color = 'r')
    
    # plt.plot( np.array((errorPeriod[ii])).T, test_data[idx[ii],errorPeriod[ii]].T,'.',color='y',\
    #               linewidth = 3, label = 'anomaly')
        
    # plt.plot( errorWhen[ii], test_data[idx[ii],int(errorWhen[ii])],'o',markersize=10, markerfacecolor='r',
    #       markeredgewidth=.5, markeredgecolor='k', label = 'start')
    plt.ylim([-4.5, 5])
    # plt.plot([i] * 10)
    camera.snap()
animation = camera.animate()
animation.save('dynamic_images2.gif')


#%%
for i in locs[:5]:
    plt.figure()
    plt.plot(test_data[i,:])
    
#%%
df.loc[12]["time"] = 200
df.loc[23:27]["time"] = 1102
df.loc[31:43]["time"] = 1356
df.loc[45:46]["time"] = 1356
df.loc[48:53]["time"] = 1356
df.loc[55:57]["time"] = 1356
df.loc[59:61]["time"] = 1356
df.loc[85]["time"] = 220
df.loc[107]["time"] = 540
#%% isolation forest

from sklearn.ensemble import IsolationForest
model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.02),max_features=1.0)
model.fit(test_data)
score =model.decision_function(test_data)
anomaly = model.predict(test_data)
locs = np.where(anomaly==-1)[0].tolist()



