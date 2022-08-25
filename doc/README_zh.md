# 介绍

## 安装依赖
```shell
pip install -r requirements.txt
```

## 修改文档内容
采用mkdocs工具构建，文档结构定义在 mkdocs.yml 文件中，文档的具体内容均在 docs 目录中。

本文档采用 markdown 语法编辑，如有新的文档需要添加，请编辑 mkdocs.yaml 添加章节即可。 

## 本地调试文档
```shell
mkdocs serve -a 127.0.0.1:8030
```     
执行上述命令后，可通过 http://127.0.0.1:8030 地址查看生成的文档内容.
> 当修改文档后，页面内容会自动更新。

## 本地构建文档
```shell
mkdocs build
```
执行上述命令后，会在 site 目录下生成文档站点的静态文件, 可将生成的静态文件进行代理访问。



