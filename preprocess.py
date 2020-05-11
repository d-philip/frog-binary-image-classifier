from keras.datasets import cifar10
from keras.utils import to_categorical
from numpy import save

class Image_Loader:

    def __init__(self):
        print("")

    def load_images(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, y_train, x_test, y_test = self.normalize_data(x_train, y_train, x_test, y_test)
        print('Cifar images preprocessed.')
        return [x_train, y_train, x_test, y_test]

    def normalize_data(self, x_train, y_train, x_test, y_test):
        # normalize images to range 0-1
        xtrain_norm = x_train.astype('float32')
        xtest_norm = x_test.astype('float32')
        xtrain_norm = xtrain_norm / 255
        xtest_norm = xtest_norm / 255
        # one-hot encode labels
        ytrain_norm = to_categorical(y_train, num_classes=10)
        ytest_norm = to_categorical(y_test, num_classes=10)
        return [xtrain_norm, ytrain_norm, xtest_norm, ytest_norm]

    def save_data(self, x_train, y_train, x_test, y_test):
        # saves training and testing data to numpy arrays
        save('x_train.npy', x_train)
        save('y_train.npy', y_train)
        save('x_test.npy', x_test)
        save('y_test.npy', y_test)
