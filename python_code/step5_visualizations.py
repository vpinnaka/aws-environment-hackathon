import matplotlib.gridspec as gridspec
import seaborn as sns

# distributions
plt.figure(figsize=(12,8*4))
gs = gridspec.GridSpec(7, 4)
for i, cn in enumerate(weekdayData_scaled.columns[:16]):
    ax = plt.subplot(gs[i])
    sns.distplot(weekdayData_scaled[cn], bins=100, label = 'train') # train data
    sns.distplot(test_data[:,96*i:96*(i+1)], bins=100, label = 'test') # test data
    sns.distplot(test_data[idx,96*i:96*(i+1)], bins=100, label = 'anomalous') # anomolous data
    ax.set_xlabel('')
    ax.set_xlim([-5, 5])
    ax.set_title('feature: ' + str(cn))
plt.legend()
plt.show()


f1 = 0
# f2 = 2
anomalies = idx
for f2 in [2,6,10,14]:
  fig, ax = plt.subplots(figsize=(10,4))
  ax.scatter(X_train[:, 96*f1:96*(f1+1):7],X_train[:, 96*f2:96*(f2+1):7], marker="s", s = 80, color="lightBlue", label = "train")
  ax.scatter(test_data[:, 96*f1:96*(f1+1):7], test_data[:, 96*f2:96*(f2+1):7], marker="o", color='Green', alpha = 0.5, label = "test")
  ax.scatter(test_data[anomalies, 96*f1:96*(f1+1)], test_data[anomalies, 96*f2:96*(f2+1)], marker ="*",color='Red', alpha = 0.5, label = "anomalous")
  
  # ax.scatter(test_data[anomalies, 96*f1:96*(f1+1)], test_data[anomalies, 96*f2:96*(f2+1)], marker ="*",color='Red', alpha = 0.5, label = "anomalous")

  plt.legend()
  plt.xlabel(weekdayData_scaled.columns[f1])
  plt.ylabel(weekdayData_scaled.columns[f2])

# for i, txt in enumerate(train_test['V14'].index):
#        if train_test_y.loc[txt] == 1 :
#             ax.annotate('*', (train_test['V14'].loc[txt],train_test['V11'].loc[txt]),fontsize=13,color='Red')
#        if predictions[i] == True :
#             ax.annotate('o', (train_test['V14'].loc[txt],train_test['V11'].loc[txt]),fontsize=15,color='Green')