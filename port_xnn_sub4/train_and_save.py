from sklearn.neural_network import MLPClassifier


from dataset_loader import load_dataset
rx_train, ry_train, rx_test, ry_test, labels, skip_ratio = load_dataset()


max_iter = 800
tol = 0.0001

model = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(), learning_rate='constant',
       learning_rate_init=0.001, max_iter=max_iter, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='lbfgs', tol=tol, validation_fraction=0.1,
       verbose=False, warm_start=False)


print("training (sub4) iter {} tol {}".format(max_iter, tol))
print("start training...")
model.fit(rx_train, ry_train)
ry_pred=model.predict(rx_test)

from s_confusion import print_confusion_report
print_confusion_report(ry_pred, ry_test, labels)

# from s_confusion import plot_confusion
# plot_confusion(ry_pred, ry_test, labels)

loc_file = __file__
from s_gen_pred import gen_pred
gen_pred(model, rx_test, skip_ratio, loc_file)
print("the training and predict data are all saved in cp_xnn subfolder, run pred to see if java output prediction is ok")
