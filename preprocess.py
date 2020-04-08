from keras.datasets import cifar10

class Image_Loader:

    def __init__(self):
        print("")

    def load_images(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('Cifar Images loaded.')

        # relabel data so that frog image label = 1 and non-frog image labels = 0
        for idx, label in enumerate(y_train):
            if label == 6:
                y_train[idx] = 1
            else:
                label = 0
        for idx, label in enumerate(y_test):
            if label == 6:
                y_test[idx] = 1
            else:
                y_test[idx] = 0
        return [x_train, y_train, x_test, y_test]
