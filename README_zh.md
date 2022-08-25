![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/license-Apache-000000.svg)
![Docs](https://github.com/iflytek/iflearner/workflows/DeployDocs/badge.svg)

# iFLearner - 一个强大且轻量的联邦学习框架
[DOCS](https://iflytek.github.io/iflearner/zh/) | [英文](https://iflytek.github.io/iflearner/)

iFLearner是一个强大且轻量的联邦学习框架，提供了一种基于数据隐私安全保护的安全计算框架，
主要针对深度学习场景下的联邦建模。其安全底层支持同态加密、秘密共享、差分隐私等多种加密技术，
算法层支持各类深度学习网络模型，并且同时支持Tensorflow、Mxnet、Pytorch等主流框架。

## 架构
![iFLeaner Arch](./doc/docs/images/iFLearner框架设计.jpg)

iFLearner主要基于以下原则进行设计:
* **事件驱动机制**: 使用事件驱动的编程范式来构建联邦学习，即将联邦学习看成是参与方之间收发消息的过程，
  通过定义消息类型以及处理消息的行为来描述联邦学习过程。
  
* **训练框架抽象**: 抽象深度学习后端，兼容支持Tensorflow、Pytorch等多类框架后端。
  
* **扩展性高**：模块式设计，用户可以自定义聚合策略，加密模块，同时支持各类场景下的算法。
  
* **轻量且简单**：该框架Lib级别，足够轻量，同时用户可以简单改造自己的深度学习算法为联邦学习算法。


## 文档
[iFLeaner Docs](https://iflytek.github.io/iflearner/zh/):
* [Installation](https://iflytek.github.io/iflearner/zh/quick_start/installation/)
* [Quickstart (TensorFlow)](https://iflytek.github.io/iflearner/zh/quick_start/quickstart_tensorflow/)
* [Quickstart (PyTorch)](https://iflytek.github.io/iflearner/zh/quick_start/quickstart_pytorch/)
* [Quickstart (MXNet)](https://iflytek.github.io/iflearner/zh/quick_start/quickstart_mxnet/)
* [Quickstart (keras)](https://iflytek.github.io/iflearner/zh/quick_start/quickstart_keras/)

## Contributor
[Contributor Guide](https://iflytek.github.io/iflearner/zh/tutorial/contributor_guide/)

## FAQ
[iFLeaner FAQ](https://iflytek.github.io/zh/iflearner/faq/faq/)

## License
[Apache License 2.0](LICENSE)
