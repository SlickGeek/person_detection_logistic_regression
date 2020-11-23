import numpy as np
import h5py


def load_dataset():
    dataset = h5py.File('../static/dataset/train_personvnonperson.hdf5', "r")
    # your train set features
    train_set_x_orig = np.array(dataset["train_set_x"][:])
    # your train set labels
    train_set_y_orig = np.array(dataset["train_set_y"][:])

    # test_dataset = h5py.File('./datasets/train_personvnonperson.hdf5', "r")
    # your test set features
    test_set_x_orig = np.array(dataset["test_set_x"][:])
    # your test set labels
    test_set_y_orig = np.array(dataset["test_set_y"][:])

    classes = np.array(dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return (train_set_x_orig, train_set_y_orig,
            test_set_x_orig, test_set_y_orig,
            classes)
