import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

def load_dataset(): 
    import s_data_loader as data_loader
    # dt = data_loader.load_feature_time()
    dt = data_loader.load_feature_time()

    # Mapping table for classes
    labels = dt.labels
    x_train = dt.x_train
    y_train = dt.y_train
    x_test = dt.x_test
    y_test = dt.y_test

    skip_ratio = 1
    rx_train = x_train[::skip_ratio]
    ry_train = y_train[::skip_ratio]
    rx_test = x_test[::skip_ratio]
    ry_test = y_test[::skip_ratio]

    return rx_train, ry_train, rx_test, ry_test, labels, skip_ratio