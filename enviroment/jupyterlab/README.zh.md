# Jupyterlab

#＃ 介绍
我们为使用 jupyterlab 构建 IFLearner 提供 Dockerfile。

## Dockerfile
- `iflearner_jupyterlab_base.Dockerfile`：仅安装 jupyterlab3.0.5 和 iflearner 库。

> 这里只提供几个简单的环境安装示例。 当然你也可以自定义Dockerfile，
> 安装 iflearner 和各种依赖项

> 如果需要使用gpu，需要从dockerhub下载一个jupyterlab的gpu版本的镜像作为基础镜像。

## 构建镜像

在当前目录下，可以指定对应的Dockerfile来构建不同的镜像，
这是构建命令的示例：
```shell
docker build -f iflearner_jupyterlab_base.Dockerfile -t iflearner_jupyterlab_base:latest .
```

## 启动jupytelab镜像
```shell
docker run -d --name iflearner_jupyterlab -p 8888:8888 -v (pwd):/opt/notebooks iflearner_jupyterlab_base:latest
```

## 访问jupyterlab
1. 首次启动获取token
```shell
docker logs iflearner_jupyterlab
```
2. 找到日志中类似http://049ac86d1ad0:8888/lab?token=0dd1ed0ee2e3ca15b5c433f57f477b8ba005a494d865f56e字段，
其中token=后面的部分为jupyterlab登录token，你也可以复制整段，输入到浏览器地址栏来直接登录。

3. 在浏览器输入http://{ip}:8888即可进行访问




