# import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
import s_data_loader as data_loader


dt = data_loader.load_feature()
# dt = data_loader.load_feature_time()
# dt = data_loader.load_feature_freq()
# dt = data_loader.load_raw_acc_x()
# dt = data_loader.load_raw_acc_z()

# Mapping table for classes
labels = dt.labels
x_train = dt.x_train
y_train = dt.y_train
x_test = dt.x_test
y_test = dt.y_test

plt.figure(figsize=(11,7))
colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
          '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

for i, r in enumerate([65,66,67,68,69,70]):
    plt.subplot(3,2,i+1)
    plt.plot(x_train[r], label=labels[y_train[r]]+str(r), color=colors[i], linewidth=2)
    plt.xlabel('Samples @50Hz')
    plt.legend(loc='upper left')
    plt.tight_layout()


plt.show()