import numpy as np
import matplotlib.pyplot as plt

file_name = 'resnet18_acc_train.txt'
fileObj=open(file_name,'r')

performance = fileObj.readlines()
acc_resnet = []
for i, line in enumerate(performance):
    accu = line.split(' | ')[-1]
    acc_resnet.append(float(accu[0:6]))

file_name = 'ChaosDonkey06_AEmodel_acc_train.txt'
fileObj=open(file_name,'r')

performance = fileObj.readlines()
acc_ChaosDonkey = []
for i, line in enumerate(performance):
    accu = line.split(' | ')[-1]
    accu.replace('/n','')
    acc_ChaosDonkey.append(float(accu[0:6]))


plt.plot(range(1,21),acc_resnet,'r',range(1,21),acc_ChaosDonkey[0:20],'k'), plt.xlabel('# Epoch'), plt.ylabel('Accuracy')
plt.legend(['Resnet18','Auto Encoder net'])


plt.show()