# 简介

**FaceX_ExpressionRecognition** 是一个基于 [FaceX](http://facex.idvxlab.com/) 的表情识别项目。

# 数据

压缩的数据文件存放在 `data` 文件夹里。将其解压为同名的文件夹后，该项目就可以运行了。由于解压后，数据文件较大，解压生成的文件夹会在版本管理中被忽略。

# 可视化

FaceX将简笔画数据处理为反应神经过程的向量形式。数据文件中，向量前两个值为运笔的方向，最后一个值表示是否提笔，`utility`中的`visualizer`方法可以对数据进行可视化，下图为一个例子：

- <img src="K_Means_SVM/figure/test.gif" alt="Example" style="zoom:150%;" />

