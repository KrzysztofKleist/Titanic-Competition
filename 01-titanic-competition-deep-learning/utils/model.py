from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler


class MyModel(keras.Model):
    def __init__(self):
        super().__init__()

        self.batch1 = layers.BatchNormalization()

        self.dense1 = layers.Dense(256, activation="relu")
        self.batch2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)

        self.dense2 = layers.Dense(256, activation="relu")
        self.batch3 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)

        self.dense3 = layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.batch1(inputs)
        x = self.dense1(x)
        x = self.batch2(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.batch3(x)
        x = self.dropout2(x)
        return self.dense3(x)


def model_init(model, lr, momentum, weight_decay):

    opt = SGD(learning_rate=lr, momentum=momentum, weight_decay=weight_decay)
    # opt = Adam()

    def my_learning_rate(epoch, lrate):
        return lrate

    lrs = LearningRateScheduler(my_learning_rate)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=15,
        min_delta=0.002,
        restore_best_weights=True,
    )

    return model, early_stopping, lrs
