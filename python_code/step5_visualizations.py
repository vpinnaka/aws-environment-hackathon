import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import numpy as np
import torch
import matplotlib.pylab as plt
import pandas as pd


test_data = joblib.load('../test_data/test_dataset_2019.numpy')
test_data = test_data.astype(np.float32)
npTrainMatrix = np.load('npTrainMatrix.npy')

orderedSensorList = ( 'co2_1','co2_2', 'co2_3', 'co2_4',                        
                      'temp_1', 'temp_2', 'temp_3', 'temp_4',                     
                      'dew_1','dew_2', 'dew_3', 'dew_4',
                      'relH_1', 'relH_2', 'relH_3', 'relH_4')

idx = pd.read_csv("submission.txt")["day"] # anomaly indexes

# distributions
plt.figure(figsize=(12,8*4))
gs = gridspec.GridSpec(7, 4)

for i, cn in enumerate(orderedSensorList):
    ax = plt.subplot(gs[i])
    sns.distplot(npTrainMatrix[:,96*i:96*(i+1)], bins=100, label = 'train') # train data
    sns.distplot(test_data[:,96*i:96*(i+1)], bins=100, label = 'test') # test data
    sns.distplot(test_data[idx,96*i:96*(i+1)], bins=100, label = 'anomalous') # anomolous data
    ax.set_xlabel('')
    ax.set_xlim([-5, 5])
    ax.set_title('feature: ' + str(cn))
plt.legend()
plt.show()

#%% scatter plot
f1 = 0
# f2 = 2
anomalies = idx
for f2 in [2,6,10,14]:
  fig, ax = plt.subplots(figsize=(10,4))
  ax.scatter(npTrainMatrix[:, 96*f1:96*(f1+1):7],npTrainMatrix[:, 96*f2:96*(f2+1):7], marker="s", s = 80, color="lightBlue", label = "train")
  ax.scatter(test_data[:, 96*f1:96*(f1+1):7], test_data[:, 96*f2:96*(f2+1):7], marker="o", color='Green', alpha = 0.5, label = "test")
  ax.scatter(test_data[anomalies, 96*f1:96*(f1+1)], test_data[anomalies, 96*f2:96*(f2+1)], marker ="*",color='Red', alpha = 0.5, label = "anomalous")
  
  # ax.scatter(test_data[anomalies, 96*f1:96*(f1+1)], test_data[anomalies, 96*f2:96*(f2+1)], marker ="*",color='Red', alpha = 0.5, label = "anomalous")

  plt.legend()
  plt.xlabel(weekdayData_scaled.columns[f1])
  plt.ylabel(weekdayData_scaled.columns[f2])

#%% joy plot
import joypy
from matplotlib import cm

labels= weekdayData_scaled.columns

fig, axes = joypy.joyplot(weekdayData_scaled, grid="y", overlap = 2,linewidth=1, legend=False, figsize=(10,8),
                          colormap=cm.autumn_r, title = "Distribution of all variables")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
for a in axes:
    a.set_xlim([-3,3]) 
    
# fig, axes = joypy.joyplot(weekdayData_scaled, by="dayIndex",  labels=labels, range_style='own', 
#                           grid="y", linewidth=1, legend=False, figsize=(6,5),
#                           title="Global daily temperature 1880-2014 \n(°C above 1950-80 average)",
#                           colormap=cm.autumn_r)

#%% time series decomposition

