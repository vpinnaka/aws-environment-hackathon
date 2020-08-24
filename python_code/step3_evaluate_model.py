import numpy as np
import torch
import matplotlib.pylab as plt
from scipy.signal import lfilter
import copy

# Create our own anomalous data set and see how the model performs
npTrainMatrix = np.load('npTrainMatrix.npy')
# MODEL_PATH = 'model.pth'
MODEL_PATH = 'model_RELU_v5.pth'
model = torch.load(MODEL_PATH)

# np.random.seed(30)
randomRows = np.random.choice(len(npTrainMatrix),200, replace = False)
labels = np.zeros(len(npTrainMatrix))
labels[randomRows] = 1


constructedData = np.random.permutation(npTrainMatrix)  # shuffled training data
constructedData2 = copy.copy(constructedData) 


# inject anomalies
for j in randomRows:
  i = np.random.choice(16)
  constructedData2[j, 96*(i):96*(i+1)] +=  2*np.random.choice([1,-1]) # simple translate up/down
  
# for j in randomRows:
#   i = np.random.choice(16)
#   constructedData2[j, 96*(i):96*(i+1)] *=  2*np.random.choice([1,-1]) # scale the subset

n = 20  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1
constructedData2_filter = lfilter(b,a,constructedData2)

# Get test MAE loss.
testingData = constructedData2.astype(np.float32)
testingData = constructedData2_filter.astype(np.float32) # filter
x_test_pred = model.forward(torch.from_numpy(testingData))

test_mae_loss = np.median(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), axis = 1)
# test_mae_loss = np.quantile(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), 0.95,axis = 1)
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

#%% precision-recall graphs
# example of a roc curve for a predictive model
from sklearn.metrics import roc_curve
from matplotlib import pyplot

pred = test_mae_loss/np.max(test_mae_loss) #np.zeros(len(npTrainMatrix))
pred[idx] = test_mae_loss[idx]/np.max(test_mae_loss)#1

# plot no skill roc curve
pyplot.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# calculate roc curve for model
fpr, tpr, _ = roc_curve(labels, pred)
# plot model roc curve
pyplot.plot(fpr, tpr, marker='.', label='AutoEncoder')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

#%%
from sklearn.metrics import precision_recall_curve
# calculate the no skill line as the proportion of the positive class
no_skill = len(labels[labels==1]) / len(labels)
# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(labels, pred)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='AutoEncoder')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

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


