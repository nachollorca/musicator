import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import os


class NeuralNetworkModel:
    def __init__(self, input_shape = 58):
        self.model = self.build_model(input_shape)
        self.input_shape = input_shape

    def build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=[input_shape]))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))
        return model

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

    def train(self, X_train, y_train, epochs, batch_size, validation_data=None, save_checkpoint=True, checkpoint_path='checkpoints'):
        if save_checkpoint:
            best_checkpoint = ModelCheckpoint(f'{checkpoint_path}/best_cnn.h5', monitor='val_loss', save_best_only=True,mode='min', verbose=1)
            last_checkpoint = ModelCheckpoint(f'{checkpoint_path}/last_cnn.h5', verbose=1)
            callbacks = [best_checkpoint, last_checkpoint]
        else:
            callbacks = []

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks
        )
        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def load_checkpoint(self, checkpoint_path = "checkpoints/last_cnn.h5"):
        if os.path.exists(checkpoint_path):
            self.model = load_model(checkpoint_path)
            return True
        else:
            return False

    def summary(self):
        self.model.summary()

    def infer(self, song: pd.DataFrame) -> pd.DataFrame:
        preds = self.model.predict(song)[0]
        cols = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
        return pd.DataFrame([preds], columns=cols, index=["prob. (%)"]).round(3)