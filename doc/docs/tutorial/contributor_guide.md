## Contributor Guide

### Prerequisites
**Python 3.7** or **above**

### Development Environment Setup
First, clone the IFLearner repository from GitHub:
```shell
$ git clone https://github.com/iflytek/iflearner.git
$ cd iflearner
```

Then, you need to create a python virtual environment using virtual tools like conda, pyenv, etc. Below is an example
of a command to create a virtual environment using conda:
```shell
$ conda create -n iflearner python==3.9
$ conda activate iflearner
```

Finally, you need to install the dependencies required by iflearner:
```shell
$ pip install -r requirements.txt
```

### Development Script
We provide some development scripts, you can find them in the ./dev directory.

###  Code Auto-Format And Auto-Test
First, execute the script to automatically format:
```shell
$ ./dev/format.sh
```
Second, execute the test script. Then, you should follow the code guidelines and make code adjustments as prompted.
```shell
$ ./dev/test.sh
```

### Build Documentation
IFLearner uses mkdocs to build documentation, you can go to the ./doc directory and follow the readme tutorial to build documentation.

### Pack Whl Release
IFLearner uses setup to pack release. You can use the following command to package:
```shell
python setup.py bdist_wheel
```
The iflearner-*.whl and iflearner-*.tar.gz packages will be stored in the ./dist subdirectory.

### Publish Release
If you have permission to publish release packages, you can do so as follows.

First, configure the pypirc file.
```shell
#Linux
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
Then, package release:
```shell
python setup.py sdist
````
Finally, to publish release:
```shell
twine upload dist/* -r pypi
````