# Jupyterlab

JupyterLab is Jupyter's latest data science production tool, and in a sense, it appeared to replace Jupyter Notebook. But don't worry about Jupyter Notebook disappearing,
JupyterLab contains all the features of Jupyter Notebook.

JupyterLab is a web-based integrated development environment, you can use it to write notebooks, operate terminals, edit markdown text, open interactive mode, view csv files and pictures and other functions.

## Example
This example mainly demonstrates how to install and run the iflearner library in jupyterlab, and start one server and two clients to complete the federated learning task of image classification under the mnist dataset.

The installation startup sequence is: 1. Server side 2. Client side.

### 1. Server side
Please refer to `pytorch_mnist_server.ipynb` for server installation and startup

### 2. Client
1. Please refer to `pytorch_mnist_client1.ipynb` to install and start client1
2. Please refer to `pytorch_mnist_client2.ipynb` to install and start client2

When both clients start to register, the federated training task will be started automatically.