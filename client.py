import flwr as fl
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# import cifar10 datasets

x_train, x_test = x_train / 255.0, x_test / 255.0

# model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
# model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    
def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        # Conv2D layer with 32 filters, 3x3 kernel size, and relu activation function
        tf.keras.layers.MaxPooling2D((2, 2)),
        # MaxPooling2D layer with 2x2 pool size
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        # Flatten layer to flatten the input
        tf.keras.layers.Dense(64, activation='relu'),
        # Dense layer with 64 units and relu activation function
        tf.keras.layers.Dense(10)
        # Dense layer with 10 units
    ])
    return model

class CifarClient(fl.client.NumPyClient):
    # Create a client class that inherits from fl.client.NumPyClient
    def __init__(self):
        self.model = create_cnn_model()
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        
    
    def get_parameters(self, config):
        # Get the current model weights
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set the given weights
        self.model.set_weights(parameters)
        self.model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        # 3 batches of 32 samples
        return self.model.get_weights(), len(x_train), {}

    # Evaluate the model using the given weights and return the loss, the number of samples, and the accuracy
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        # Evaluate the model using the given weights
        return loss, len(x_test), {"accuracy": float(accuracy)}

    
fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())