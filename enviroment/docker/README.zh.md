# Docker

#＃ 介绍
我们为 IFLearner 提供 Dockerfiles。
需要先安装docker，如果需要使用gpu训练，请先安装nvidia驱动和nvidia-docker。

#＃ Dockerfile
- `iflearner_base.Dockerfile`：仅安装 iflearner 库。
- `iflearner_torch1.7.1.Dockerfile`：安装iflearner、torch1.7.1、torchvision0.8.2库。
- `iflearner_tensorflow2.9.1.Dockerfile`：安装iflearner、tensorflow2.9.1库。
- `iflearner_mxnet1.9.1.Dockerfile`：安装iflearner、mxnet1.9.1库。
- `iflearner_kera2.9.0.Dockerfile`：安装iflearner、keras2.9.0库。

> 这里只提供几个简单的环境安装示例。当然你也可以自定义Dockerfile， 安装 iflearner 和各种依赖项.
> 如果你需要使用gpu, 则需要在镜像中安装对应的cuda或者从dockerhub下载一个cuda版本镜像作为基础镜像。

## 构建镜像
在当前目录下，可以指定对应的Dockerfile来构建不同的镜像， 下面是一个构建示例：
```shell
docker build -f iflearner_torch1.7.1.Dockerfile -t iflearner_torch1.7.1 .
```