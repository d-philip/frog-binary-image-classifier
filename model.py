from preprocess import Image_Loader
from tensorflow.keras import Sequential, layers
from keras.preprocessing.image import load_img
import tensorflow as tf
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt
from numpy import asarray


class CNNModel:

    def __init__(self):
        self.x_train = self.x_test = self.y_train = self.y_test = None
        self.history = None
        self.model = None
        self.model_dir = 'models/'

    def load_data(self):
        IMG = Image_Loader()
        [self.x_train, self.y_train, self.x_test, self.y_test] = IMG.load_images()
        print('Training and testing data loaded.')

    def build_model(self):
        x_shape = self.x_train[0].shape
        self.model = Sequential()
        self.model.add(layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        self.model.add(layers.MaxPooling2D(2, 2))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.model.add(layers.Dense(10, activation='sigmoid'))
        self.model.add(layers.Dropout(0.2))
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
        self.model.summary()

    def train(self):
        num_epochs = 100
        num_batches = 64
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=num_batches, epochs=num_epochs, validation_data=(self.x_test, self.y_test), callbacks=[TqdmCallback()])

    def eval_model(self):
        try:
            score = self.model.evaluate(self.x_train, self.y_train)
            print("Training Loss: ", score[0])
            print("Training Accuracy: ", score[1])
            score = self.model.evaluate(self.x_test, self.y_test)
            print("Testing Loss: ", score[0])
            print("Testing Accuracy: ", score[1])

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
        img = load_img(image, target_size=(32,32))
        img_pix = asarray(img)
        img_pix = img_pix.reshape(1, 32, 32, 3)
        img_pix = img_pix.astype('float32')
        img_pix = img_pix / 255.0
        print('Image loaded.')
        return img_pix

    def predict(self, image):
        img_pix = self.load_image(image)
        y_pred = self.model.predict_classes(img_pix)
        return y_pred

    def serialize(self, filename):
        file = self.model_dir + filename
        self.model.save(file, save_format='h5')
        print('Model saved.')

    def deserialize(self, filename):
        file = self.model_dir + filename
        model = tf.keras.models.load_model(file)
        print('Model loaded.')
        return model
