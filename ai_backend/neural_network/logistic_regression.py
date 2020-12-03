from ai_backend.utils.lr_utils import load_dataset
from ai_backend.neural_network.model import model


def logistic_regression():
    # Loading the data (person/non-person)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    w, b = model(train_set_x, train_set_y, test_set_x, test_set_y,
                 num_iterations=2000, learning_rate=0.005, print_cost=True)

    return w, b
