## 贡献者指南

### 先决条件
**Python 3.7** 或 **以上**

### 开发环境设置
首先，从 GitHub 克隆 IFLearner 库：
```shell
$ git clone https://github.com/iflytek/iflearner.git
$ cd iflearner
```

然后，您需要使用 conda、pyenv 等虚拟工具创建一个 python 虚拟环境。下面是一个使用 conda 创建虚拟环境的示例：
```shell
$ conda create -n iflearner python==3.9
$ conda activate iflearner
```

最后，需要安装iflearner所需的依赖：
```shell
$ pip install -r requirements.txt
```

### 开发脚本
我们提供了一些开发脚本，您可以在 ./dev 目录中找到它们。

### 代码自动格式化和自动测试
首先，执行脚本自动格式化：
```shell
$ ./dev/format.sh
```
其次，执行测试脚本。然后，您应该遵循代码规范，根据提示进行代码调整。
```shell
$ ./dev/test.sh
```

### 构建文档
IFLearner 使用 mkdocs 构建文档，您可以进入 ./doc 目录并按照readme教程构建文档。

### Whl打包
IFLearner 使用 setup 来进行打包：
```shell
python setup.py bdist_wheel
```
iflearner-*.whl 和 iflearner-*.tar.gz 包将存储在 ./dist 子目录中。

### 发布Pypi
如果你有权限发布版本包，你可以按下述操作进行发布.

首先, 配置pypirc文件.
```shell
# Linux
## vim ~/.pypirc
# Windows
## C:\Users\Username\.pypirc
: <<'COMMENT'
 [distutils]
 index-servers=pypi

 [pypi]
 repository=https://upload.pypi.org/legacy/
 username=<username>
 password=<password>
COMMENT
```
然后, 进行打包:
```shell
python setup.py sdist
```
最后, 进行发布:
```shell
twine pload dist/* -r pypi
```