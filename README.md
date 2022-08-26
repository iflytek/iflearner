# iFLearner - A Powerful and Lightweight Federated Learning Framework
![](https://img.shields.io/badge/language-python-blue.svg)
[![Forks](https://img.shields.io/github/forks/iflytek/iflearner)](https://img.shields.io/github/forks/iflytek/iflearner)
[![Stars](https://img.shields.io/github/stars/iflytek/iflearner)](https://img.shields.io/github/stars/iflytek/iflearner)
[![Docs](https://github.com/iflytek/iflearner/actions/workflows/deploy_doc.yaml/badge.svg)](https://github.com/iflytek/iflearner/actions/workflows/deploy_doc.yaml)
[![Pypi](https://github.com/iflytek/iflearner/actions/workflows/publish_pypi.yaml/badge.svg)](https://github.com/iflytek/iflearner/actions/workflows/publish_pypi.yaml)
[![Contributors](https://img.shields.io/github/contributors/iflytek/iflearner)](https://github.com/iflytek/iflearner/graphs/contributors)
[![License: Apache2.0](https://img.shields.io/github/license/iflytek/iflearner)](https://github.com/iflytek/iflearner/blob/main/LICENSE)

[DOCS](https://iflytek.github.io/iflearner/) | [中文](https://iflytek.github.io/iflearner/zh/)

iFLearner is a federated learning framework, which provides a secure computing framework based on 
data privacy security protection, mainly for federated modeling in deep learning scenarios. Its security bottom 
layer supports various encryption technologies such as homomorphic encryption, secret sharing, and differential 
privacy. The algorithm layer supports various deep learning network models, and supports mainstream frameworks 
such as Tensorflow, Mxnet, and Pytorch. 
 
## Architecture
![iFLeaner Arch](./doc/docs/images/iFLearner框架设计.jpg)

The design of iFLearner is based on a few guiding principles:

* **Event-driven mechanism**: Use an event-driven programming paradigm to build federated learning, that is, 
  to regard federated learning as the process of sending and receiving messages between participants,
  and describe the federated learning process by defining message types and the behavior of processing messages.
  
* **Training framework abstraction**: Abstract deep learning backend, compatible with support for multiple 
  types of framework backends such as Tensorflow and Pytorch.
  
* **High scalability: modular design**, users can customize aggregation strategies, encryption modules,
  and support algorithms in various scenarios.
  
* **Lightweight and simple**: The framework is Lib level, light enough, and users can simply transform their deep 
  learning algorithms into federated learning algorithms.
  
## Documentation
[iFLeaner Docs](https://iflytek.github.io/iflearner/):
* [Installation](https://iflytek.github.io/iflearner/quick_start/installation/)
* [Quickstart (TensorFlow)](https://iflytek.github.io/iflearner/quick_start/quickstart_tensorflow/)
* [Quickstart (PyTorch)](https://iflytek.github.io/iflearner/quick_start/quickstart_pytorch/)
* [Quickstart (MXNet)](https://iflytek.github.io/iflearner/quick_start/quickstart_mxnet/)
* [Quickstart (keras)](https://iflytek.github.io/iflearner/quick_start/quickstart_keras/)
* [Quickstart (DP-Opacus)](https://iflytek.github.io/iflearner/quick_start/quickstart_smpc/)
* [Quickstart (SMPC)](https://iflytek.github.io/iflearner/quick_start/quickstart_opacus/)

## Contributor
[Contributor Guide](https://iflytek.github.io/iflearner/tutorial/contributor_guide/)

## FAQ
[iFLeaner FAQ](https://iflytek.github.io/iflearner/faq/faq/)

## License
[Apache License 2.0](LICENSE)

