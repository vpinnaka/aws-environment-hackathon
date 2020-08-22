import numpy as np
import torch
import matplotlib.pylab as plt
import joblib


MODEL_PATH = 'model.pth'
# MODEL_PATH = 'model_RELU.pth'
model = torch.load(MODEL_PATH)


test_data = joblib.load('../test_data/test_dataset_2019.numpy')
testingData = test_data.astype(np.float32)

x_test_pred = model.forward(torch.from_numpy(testingData))

# test_mae_loss = np.mean(np.abs(testingData[:,:16*96] - x_test_pred[:,:16*96].detach().numpy()), axis = 1)
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
for i in range(5):
  plt.figure(figsize=(10, 3))
  # plt.plot(test_data[idx[i], :])
  plt.plot(testingData[idx[i], :],color='r')
  plt.plot(x_test_pred.detach().numpy()[idx[i],:])
  
# find where anomaly starts  
  
#%% Generate output
df = pd.DataFrame(idx[:200])