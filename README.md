# ai_learning

## 项目介绍
> 机器学习和深度学习相关书籍及书籍自带源码和本人在学习过程中做的一些笔记

## 使用方法

- `git clone https://github.com/chengyingshe/ai_learning.git`：克隆仓库到本地
- `git switch <brach_name>`：切换分支
- 推荐学习顺序：`dl` -> `ml` -> `dl_on_action`

## 分支介绍

- `ml` ：python手写机器学习

  > 这部分我提供的笔记中使用了现成的`sklearn`库函数低代码实现相同的算法功能

- `dl` ：python手写深度学习

  > 这部分我提供的笔记中使用了现成的`pytorch`库函数低代码实现简单模型的搭建和训练

- `dl_on_action` ：深度学习实战

  > 这部分书中主要是使用`keras`深度学习框架实现一些经典案例，强烈建议在学习完前两个分支里面的内容后，对深度学习和传统的机器学习有一定的了解之后再学习这部分
  >
  > 这部分我的笔记也是比较详细的​，个人认为还是比较有参考价值:laughing:

  > 部分人在运行书中的源码时可能会出现`tensorflow`报错，这个跟`tensorflow`的版本有关，推荐下载`2.10.0`版本，并且需要在代码的最开始添加（这部分可以参照我的笔记）：
  >
  > ```python
  > import tensorflow as tf
  > tf.compat.v1.disable_eager_execution()
  > ```

> 每个分支中都是相同的结构：
> - `pdf` ：包含书籍的pdf文件（声明：仅用于学习，请勿进行传播）
> - `code` ：书中自带的源码
> - `notebook` ：本人在学习过程中做的一些笔记（一般都是一个jupyter notebook，可以使用单元格进行运行调试）