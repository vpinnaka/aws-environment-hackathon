# Create our own anomalous data set and see how the model performs

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
  # constructedData2[j, 96*(i):96*(i+1)] *=  2*np.random.choice([1,-1])


# Get test MAE loss.
testingData = constructedData2.astype(np.float32)
# testingData = constructedData2_filter.astype(np.float32)
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