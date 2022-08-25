## Quickstart (TensorFlow)

In this tutorial, we will describe how to use Ifleaner in the Tensorflow framework to complete image classification federated training on the MNIST dataset.

our example contains two clients and one server by default. In each round of training, 
the client is responsible for training and uploading the model parameters to the server, the server
aggregates, and sends the aggregated global model parameters to each client, and then each client 
updates the aggregated model parameters. Multiple rounds will be repeated.

First of all, we highly recommend to create a python virtual environment to run, you can use virtual tools such as virtualenv, pyenv, conda, etc.

Next, we can quickly install the IFLearner library with the following command:
```shell
pip install iflearner
````

Also, since we want to use Tensorflow for image classification tasks on MNIST data, we need to go ahead and install the Tensorflow libraries:
```shell
pip install tensorflow==2.9.1
````

### Ifleaner Server

1.  Create a new file named `server.py`, import iflearner:
```python
from iflearner.business.homo.aggregate_server import main

if __name__ == "__main__":
    main()
```

2. You can start the server with the follow command:
```shell
python server.py -n 2
```
> -n 2: Receive two clients for federated training


### Ifleaner Client
Create a new file named `quickstart_tensorflow.py` and do the following.

#### 1. Define Model Network

Firstly, you need define your model network by using TensorFlow.

```python
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)
```

#### 2. Implement Trainer Class

Secondly, you need implement your trainer class, inheriting from the `iflearner.business.homo.trainer.Trainer` class. 
The class need to implement four functions, which are `get`, `set`, `fit` and `evaluate`. 
We also have provided a `iflearner.business.homo.tensorflow_trainer.TensorFlowTrainer` inheriting from the `iflearner.business.homo.trainer.Trainer` class,
which has implement usual `get` and `set` functions.

You can use this class as the follow:

```python
import tensorflow as tf
from iflearner.business.homo.tensorflow_trainer import TensorFlowTrainer
from iflearner.business.homo.train_client import Controller

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

class Mnist(TensorFlowTrainer):
    def __init__(self, model) -> None:
        super().__init__(model)

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    def fit(self, epoch):
        for images, labels in train_ds:
            # train_step(images, labels)
            self._fit(images, labels)           

    @tf.function
    def _fit(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    def evaluate(self, epoch):
        for test_images, test_labels in test_ds:
            # test_step(test_images, test_labels)
            self._evaluate(test_images, test_labels)

        print(
            f'Epoch {epoch}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
        return {'Accuracy': train_accuracy.result() * 100}

    @tf.function
    def _evaluate(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
```

#### 3. Start Ifleaner Client

Lastly, you need to write a `main` function to start your client. 

You can do it as the follow:

```python
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model = MyModel()
    mnist = Mnist(model)
    controller = Controller(args, mnist)
    controller.run()
```

In the `main` function, you need import `parser` from `iflearner.business.homo.argument` firstly and then call `parser.parse_args`, 
because we provided some common arguments that need to be parsered. If you want to add addtional arguments for yourself, you can call `parser.add_argument` repeatedly to add them before 
`parser.parse_args` has been called. After parsered arguments, you can create your trainer instance base on previous implemented class, and put it with `args` to `train_client.Controller`.
In the end, you just need call `controller.run` to run your client.

You can use follow command to start the first client:
```shell
python quickstart_tensorflow.py --name client01 --epochs 2
```

Open another terminal and start the second client:
```shell
python quickstart_tensorflow.py --name client02 --epochs 2
```

After both clients are ready and started, we can see log messages similar to the following on either client terminal:
```text
2022-08-03 18:07:07.406604: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Namespace(name='client01', epochs=2, server='localhost:50001', enable_ll=0, peers=None, cert=None)
2022-08-03 18:07:07.456 | INFO     | iflearner.business.homo.train_client:run:90 - register to server
2022-08-03 18:07:07.479 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_register, time: 22.46353199999973ms
2022-08-03 18:07:07.479 | INFO     | iflearner.business.homo.train_client:run:107 - use strategy: FedAvg
2022-08-03 18:07:07.480 | INFO     | iflearner.business.homo.train_client:run:140 - report client ready
2022-08-03 18:07:07.482 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.3502309999999795ms
2022-08-03 18:07:11.500 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 1.1013739999992112ms
2022-08-03 18:07:12.497 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
2022-08-03 18:08:19.804 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
Epoch 1, Loss: 0.1414850801229477, Accuracy: 95.73500061035156, Test Loss: 0.0603780597448349, Test Accuracy: 0.0
2022-08-03 18:08:22.759 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
2022-08-03 18:08:24.130 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_upload_param, time: 149.0828160000035ms
2022-08-03 18:08:31.445 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_aggregate_result, time: 3241.557815999997ms
2022-08-03 18:08:32.446 | INFO     | iflearner.business.homo.train_client:run:222 - ----- set -----
2022-08-03 18:08:32.474 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_client_ready, time: 1.617487000004303ms
2022-08-03 18:08:33.469 | INFO     | iflearner.communication.homo.homo_client:notice:94 - IN: party: message type: msg_notify_training, time: 8.709660999997482ms
2022-08-03 18:08:33.479 | INFO     | iflearner.business.homo.train_client:run:150 - ----- fit <FT> -----
2022-08-03 18:09:46.374 | INFO     | iflearner.business.homo.train_client:run:168 - ----- evaluate <FT> -----
Epoch 2, Loss: 0.11132627725601196, Accuracy: 96.65583038330078, Test Loss: 0.0679144412279129, Test Accuracy: 0.0
2022-08-03 18:09:49.340 | INFO     | iflearner.business.homo.train_client:run:179 - ----- get <FT> -----
label: FT, points: ([1, 2], [<tf.Tensor: shape=(), dtype=float32, numpy=95.735>, <tf.Tensor: shape=(), dtype=float32, numpy=96.65583>])
label: LT, points: ([1], [<tf.Tensor: shape=(), dtype=float32, numpy=95.735>])
2022-08-03 18:09:51.374 | INFO     | iflearner.communication.homo.homo_client:transport:59 - OUT: message type: msg_complete, time: 1.561914999996361ms
```
congratulations! You have successfully built and run your first federated learning system. The complete source code 
reference for this example [Quickstart_Tensorflow](https://github.com/iflytek/iflearner/tree/main/examples/homo/quickstart_tensorflow).