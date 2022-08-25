# Jupyterlab

## Introduction
We provide Dockerfiles for build IFLearner with jupyterlab.

## Dockerfile
- `iflearner_jupyterlab_base.Dockerfile`: Only install jupyterlab3.0.5 and iflearner library.

> Here, we just provide a few simple environment installation examples. Of course, you can also customize the Dockerfile, 
> install iflearner and various dependencies

> If you need to use gpu, you need to install a jupyterlab gpu image from dockerhub as the base image.

## Build Image

In the current directory, you can specify the corresponding Dockerfile to build different images,
here is an example of a build command:
```shell
docker build -f iflearner_jupyterlab_base.Dockerfile  -t  iflearner_jupyterlab_base:latest .
```

## Start the jupytelab image
```shell
docker run -d --name iflearner_jupyterlab -p 8888:8888 -v ${pwd}:/opt/notebooks iflearner_jupyterlab_base:latest
````

## Visit jupyterlab
1. First start to get token
```shell
docker logs iflearner_jupyterlab
````
2. Find a field similar to http://049ac86d1ad0:8888/lab?token=0dd1ed0ee2e3ca15b5c433f57f477b8ba005a494d865f56e in the log,
The part after token= is the jupyterlab login token. You can also copy the entire paragraph and enter it into the browser address bar to log in directly.

3. Enter http://{ip}:8888 in the browser to access
