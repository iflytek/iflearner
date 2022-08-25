import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from iflearner.business.homo.argument import parser
from iflearner.business.homo.tensorflow_trainer import TensorFlowTrainer
from iflearner.business.homo.train_client import Controller


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")


class Mnist(TensorFlowTrainer):
    def __init__(self, model) -> None:
        super().__init__(model)

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    def fit(self, epoch):
        for images, labels in train_ds:
            self._fit(images, labels)

    @tf.function
    def _fit(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    def evaluate(self, epoch):
        for test_images, test_labels in test_ds:
            self._evaluate(test_images, test_labels)

        print(
            f"Epoch {epoch}, "
            f"Loss: {train_loss.result()}, "
            f"Accuracy: {train_accuracy.result() * 100}, "
            f"Test Loss: {test_loss.result()}, "
            f"Test Accuracy: {test_accuracy.result() * 100}"
        )
        return {"Accuracy": train_accuracy.result() * 100}

    @tf.function
    def _evaluate(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model = MyModel()
    mnist = Mnist(model)
    controller = Controller(args, mnist)
    controller.run()
