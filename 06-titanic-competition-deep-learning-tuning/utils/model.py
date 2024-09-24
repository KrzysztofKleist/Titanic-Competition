from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import regularizers


class MyModel(keras.Model):
    def __init__(self, l2=0.01, dropout=0.3):
        super().__init__()

        self.batch1 = layers.BatchNormalization()

        # Add L2 regularization to the Dense layers
        self.dense1 = layers.Dense(
            256, activation="relu", kernel_regularizer=regularizers.L2(l2)
        )
        self.batch2 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout)

        self.dense2 = layers.Dense(
            256, activation="relu", kernel_regularizer=regularizers.L2(l2)
        )
        self.batch3 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout)

        self.dense3 = layers.Dense(
            1, activation="sigmoid", kernel_regularizer=regularizers.L2(l2)
        )

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

    # # Define a custom learning rate schedule function
    # def lr_schedule(epoch, lr):
    #     if epoch % 8 == 0 and epoch > 0:
    #         return lr * 0.5  # Reduce learning rate by half every 10 epochs
    #     return lr  # Otherwise, keep the learning rate the same

    # lrs = LearningRateScheduler(lr_schedule)

    # Define the ReduceLROnPlateau callback
    lrs = ReduceLROnPlateau(
        monitor="val_loss",  # metric to monitor (can also be 'val_accuracy', etc.)
        factor=0.5,  # factor by which the learning rate will be reduced
        patience=5,  # number of epochs with no improvement after which learning rate will be reduced
        min_lr=1e-5,  # minimum learning rate
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=15,
        min_delta=0.002,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    return model, early_stopping, lrs
