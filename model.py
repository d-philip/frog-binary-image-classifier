from preprocess import Image_Loader
from tensorflow.keras import Sequential, layers
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt

IMG = Image_Loader()

class CNNModel:

    def __init__(self):
        [self.images, self.labels] = IMG.load_images()
        [self.x_train, self.x_test, self.y_train, self.y_test] = IMG.split_data(self.images, self.labels)
        self.model=self.build_model()

    def build_model(self):
        x_shape = self.images[0].shape
        model = Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
        model.add(layers.MaxPooling2D(2, 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
        return model

    def train(self):
        # TODO: pickle the model
        num_epochs = 20
        num_batches = 32
        history = self.model.fit(self.x_train, self.y_train, batch_size=num_batches, epochs=num_epochs, validation_data=(self.x_test, self.y_test), callbacks=[TqdmCallback()])
        return model

    def eval_model(self, model=None):
        score = []
        score[0] = model.evaluate(self.x_train, self.y_train)
        print("Training Accuracy: ", score[0])
        score[1] = model.evaluate(self.x_test, self.y_test)
        print("Testing Accuracy: ", score[1])

        plt.subplot(1, 2, 1)
        plt.plot(history.history['acc'], label='acc')
        plt.plot(history.history['val_acc'], label = 'val_acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='upper right')

    def predict(self):
        # TODO: Implement model prediction functionality
        print(hello)
