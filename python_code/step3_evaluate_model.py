import numpy as np
import torch
import matplotlib.pylab as plt


# Create our own anomalous data set and see how the model performs
npTrainMatrix = np.load('npTrainMatrix.npy')
# MODEL_PATH = 'model.pth'
MODEL_PATH = 'model_RELU.pth'
model = torch.load(MODEL_PATH)

# np.random.seed(30)
randomRows = np.random.choice(len(npTrainMatrix),200, replace = False)
labels = np.zeros(len(npTrainMatrix))
labels[randomRows] = 1

import copy
constructedData = np.random.permutation(npTrainMatrix)  # shuffled training data
constructedData2 = copy.copy(constructedData) 


# inject anomalies
for j in randomRows:
  i = np.random.choice(16)
  constructedData2[j, 96*(i):96*(i+1)] +=  2*np.random.choice([1,-1]) # simple translate up/down
  
# for j in randomRows:
#   i = np.random.choice(16)
#   constructedData2[j, 96*(i):96*(i+1)] *=  2*np.random.choice([1,-1]) # scale the subset


# Get test MAE loss.
testingData = constructedData2.astype(np.float32)
x_test_pred = model.forward(torch.from_numpy(testingData))

# test_mae_loss = np.mean(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), axis = 1)
test_mae_loss = np.quantile(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), 0.95,axis = 1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.figure(figsize=(6,3)) 
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
# threshold = 0.45
# anomalies = (test_mae_loss > threshold).tolist()
# print("Number of anomaly samples: ", np.sum(anomalies))
# print("Indices of anomaly samples: ", np.where(anomalies))

idx = (-test_mae_loss).argsort()[:200] # predicted windows where anomaly exists
pred = np.zeros(len(npTrainMatrix))
pred[idx] = 1

#%%
TP = 0
FN = 0
FNs = []
TPs = []
for i in randomRows:
  if pred[i] == 1:
    TP += 1
    TPs.append(i)
  else:
    FN += 1
    FNs.append(i)
    
TN = 0
FP = 0
FPs = []
TNs = []
idx_neg = [curr for curr,i in enumerate(labels) if i == 0]

for i in idx_neg:
  if pred[i] == 0:
    TN += 1
    TNs.append(i)
  else:
    FP += 1  
    FPs.append(i)   
    
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(labels, pred) # This score corresponds to the area under the precision-recall curve.

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))
    

from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(labels, pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))
print(TP/(TP+FP))
print('TP = {0}, TN = {1}, FP = {2}, FN = {3}'.format(TP, TN, FP, FN))
#%% visualize examples
samples = [FNs, FPs, TNs, TPs]
legends = ["FNs","FPs","TNs","TPs"]

j = 0
for s in samples:
    fig, ax = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(10, 6))
    
    for i in range(6,12):
      # plt.figure(figsize=(10, 3))
      i = i % 6
      plt.subplot(3, 2, i+1)
      plt.plot(testingData[s[i], :])
      plt.plot(x_test_pred.detach().numpy()[s[i],:])
    
      plt.title(legends[j])
    j += 1


