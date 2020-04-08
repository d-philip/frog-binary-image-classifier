from preprocess import Image_Loader
from tensorflow.keras import Sequential, layers
from keras.preprocessing.image import load_img
import tensorflow as tf
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
from numpy import asarray


class CNNModel:

    def __init__(self):
        self.images = self.labels = None
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.history = None
        self.model = None

    def load_data(self):
        IMG = Image_Loader()
        [self.images, self.labels] = IMG.load_images()
        [self.x_train, self.x_test, self.y_train, self.y_test] = IMG.split_data(self.images, self.labels)

    def build_model(self):
        x_shape = self.images[0].shape
        self.model = Sequential()
        self.model.add(layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
        self.model.summary()

    def train(self):
        num_epochs = 20
        num_batches = 32
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=num_batches, epochs=num_epochs, validation_data=(self.x_test, self.y_test), callbacks=[TqdmCallback()])

    def eval_model(self):
        try:
            score = self.model.evaluate(self.x_train, self.y_train)
            print("Training Accuracy: ", score)
            score = self.model.evaluate(self.x_test, self.y_test)
            print("Testing Accuracy: ", score)

            if (self.history):
                plt.subplot(1, 2, 1)
                plt.plot(self.history.history['accuracy'], label='accuracy')
                plt.plot(self.history.history['val_accuracy'], label = 'val_accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.ylim([0.7, 1])
                plt.legend(loc='lower right')

                plt.subplot(1, 2, 2)
                plt.plot(self.history.history['loss'], label='loss')
                plt.plot(self.history.history['val_loss'], label = 'val_loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.ylim([0.7, 1])
                plt.legend(loc='upper right')
        except:
            if (self.x_train is None):
                print('Train and test data not loaded. Run CNNModel.load_data().')
            else:
                print('Please make sure train and test data are loaded correctly.')

    def load_image(self, image):
        img = load_img(image, target_size=(64, 64))
        img_pix = asarray(img)
        img_pix = img_pix.reshape(1, 64, 64, 3)
        print('Image loaded')
        return img_pix

    def predict(self, img_pix):
        y_pred = self.model.predict_classes(img_pix)
        return y_pred

    def serialize(self, filename):
        self.model.save(filename, save_format='h5')

    def deserialize(self, filename):
        model = tf.keras.models.load_model(filename)
        return model
