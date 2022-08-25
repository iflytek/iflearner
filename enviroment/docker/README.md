# Docker

## Introduction
We provide Dockerfiles for IFLearner.
You need to install docker first, and if you need to use gpu training, please install nvidia driver and nvidia-docker first.

## Dockerfile
- `iflearner_base.Dockerfile`: Only install iflearner library.
- `iflearner_torch1.7.1.Dockerfile`:  Install iflearner、torch1.7.1、torchvision0.8.2 library.
- `iflearner_tensorflow2.9.1.Dockerfile`:  Install iflearner、tensorflow2.9.1 library.
- `iflearner_mxnet1.9.1.Dockerfile`:  Install iflearner、mxnet1.9.1 library.
- `iflearner_kera2.9.0.Dockerfile`:  Install iflearner、keras2.9.0 library.

> Here, we just provide a few simple environment installation examples. Of course, you can also customize the Dockerfile, 
> install iflearner and various dependencies

> If you need to use gpu, you need to install the corresponding cuda in the image or download a cuda version image from 
> dockerhub as the base image.

## Build Image

In the current directory, you can specify the corresponding Dockerfile to build different images,
here is an example of a build command:
```shell
docker build -f iflearner_torch1.7.1.Dockerfile  -t  iflearner_torch1.7.1 .
```