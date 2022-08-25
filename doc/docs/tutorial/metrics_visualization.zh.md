## 指标可视化

我们在iflearner中集成了[VisualDL](https://github.com/PaddlePaddle/VisualDL)来可视化训练指标，当你完成训练的时候，在你的训练代码目录下会有一个名为metric的目录，可以通过以下的命令来启动[VisualDL](https://github.com/PaddlePaddle/VisualDL)：

```
visualdl --logdir ./metric --host 127.0.0.1 --port 8082
```

然后，你可以在浏览器中打开链接 <http://127.0.0.1:8082>。

![VisualDL](docs/../../images/visualdl.png)
