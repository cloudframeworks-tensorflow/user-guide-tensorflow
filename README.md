# [云框架]TensorFlow

![](https://img.shields.io/badge/Release-v1.5-green.svg)
[![](https://img.shields.io/badge/Producer-yscing-orange.svg)](CONTRIBUTORS.md)
![](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

[TensorFlow](https://www.tensorflow.org/)是Google开源的人工智能（Machine Intelligence）软件库，更具体来说，TensorFlow是使用数据流图（[Data Flow Graphs](http://web.cecs.pdx.edu/~mperkows/temp/JULY/data-flow-graph.pdf)）进行数值计算的开源软件库。**Tensor**（[张量](https://en.wikipedia.org/wiki/Tensor)）指的是在节点间相互联系的多维数据数组，**Flow**（流）指基于数据流图的计算。TensorFlow架构灵活，小到智能手机，大到数据中心服务器均可展开计算，同时具备很强的通用性，适用于包括图形分类、音频处理、推荐系统和自然语言处理等在内的各种计算领域。

TensorFlow具备以下特点——

* 灵活（Deep Flexibility）：支持任何数据流图计算
* 便携（True Portability）：支持各类计算设备
* 链接科研和产品（Connect Research and Product）：加速研究成果转化为实际产品
* 自动化微分运算（Auto-Differentiation）：帮助机器学习算法自动求出梯度
* 多语言（Language Options）：利用python构建和执行计算图，支持C++的语言，未来将支持Lua、JavaScript、R等
* 性能优化（Maximize Performance）：支持线程、队列、异步计算，并根据需要分配计算元素

自15年年底开源以来，TensorFlow迅速流行，除了Google自己，Airbnb、Snapchat、eBay、Twitter等知名公司纷纷加入到TensorFlow的使用者阵营当中。

本篇[云框架](ABOUT.md)将以**ErGo**（一款基于TensorFlow的Chatbot）为例介绍TensorFlow实践。

# 内容概览

* [快速部署](#快速部署)
    * [一键部署](#一键部署)
    * [本地部署](#本地部署) 
* [业务说明](#业务说明)
* [技术流程](#技术流程)
    * [初始数据单元](#初始数据单元)
    * [数据加工](#数据加工)
    * [训练后数据](#训练后数据)
* [更新计划](#更新计划)
* [社群贡献](#社群贡献)

# <a name="快速部署"></a>快速部署

## 一键部署

[一键部署在好雨云平台](http://app.goodrain.com/group/detail/30/)

## 本地部署

1. [准备Docker环境](./READMORE/install-docker.md)
 
2. Git clone

    ```
    git clone https://github.com/cloudframeworks-tensorflow/ErGo
    ```

3. 执行如下命令，进行训练

    训练时长主要取决于数据大小、learningRate（学习效率）、[dropout](https://www.tensorflow.org/get_started/mnist/pros)及设备计算能力（推荐使用GPU，训练完成后脚本会自动退出）

    ```
    cd ErGo
    python main.py --train
    ```

4. 初始化web ui

    ```python
    docker build -t ergo -f Dockerfile.cpu .
    redis-server &
    docker run -itd -p 8000:8000 --name ergo ergo
    ```

5. 访问

    ```
    http://localhost:8000
    ```

# <a name="业务说明"></a>业务说明

聊天机器人（Chatbot）——**ErGo**，基于TensorFlow实现，可与用户互动完成智能对话。

工作流程可分为提问（Ask）、检索（Retrieve）、抽取（Extraction）、回答（Answer）4部分，用户通过界面（Web）提出问句，ErGo将在已训练数据（Trained Data）中检索并抽取答案，通过界面反馈给用户，如下图所示：

<div align=center><img width="900" height="" src="./image/work-flow.png"/></div>

例如——

```
Master: HI ERGO
ERGO: HI

Master: I LOVE YOU
ERGO: I'M SORRY
```

# <a name="技术流程"></a>ErGO实现

## 整体架构
ErGo的整体架构如下图所示：

<div align=center><img width="900" height="" src="./image/ergo-flow.png"/></div>



* ErGo加载Data（语料）并进行数据处理
* 处理完成后，由训练模型（Training Model）加载并进行反复训练
* 完成训练后，ErGo可根据训练后的数据进行相关预测，即与用户完成对话

## 模型选取
聊天机器人使用的深度学习模型几乎都是Seq2Seq,ErGo也是如此。Seq2Seq模型由两个主要部件组成，一个是编码器RNN，另一个是解码器RNN。从高层次上来说，编码器的工作是将输入文本信息生成固定的表示。解码器则是接收这个表示，并生成一个可变长度的文本，以响应它。

可以参考
- [seq2seq模型初探](https://github.com/zhuanxuhit/nd101/blob/master/1.Intro_to_Deep_Learning/11.How_to_Make_a_Language_Translator/1-seq2seq.ipynb)
- [seq2seq model](https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/seq2seq_model.py)

## <a name="初始数据单元"></a>数据集的选择

在考虑到训练机器人来回答具体某类问题或提供某种服务,第一件事就到选择哪种数据集，然后基于此进行我们需要的模型进行训练。对于序列模型，我们需要大量的会话日志。从高层次上讲，这个编码器-解码器网络需要能够正确理解每个查询(编码器输入)所期望的响应类型(解码器输出)。
常见的数据集有康奈尔电影对话语料库、Ubuntu对话语料库等。
本例中所使用的数据集来源于[Cornell_Movie-Dialogs_Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)


## 创建数据集

数据集的创建是机器学习的一个重要组成部分，它涉及到数据集预处理.
本例使用的是已经预处理完的数据,如果想研究如何预处理数据，可以研究[Word2vec](https://github.com/dav/word2vec)

## <a name="数据加工"></a>训练

1. Tensorflow原生支持多种数据读取方式，本例默认使用**从文件中读取**的方式加载处理初始数据，处理后的数据会保存为随机生成的pkl文件
2. 本例中数据模型基于循环神经网络（[RNN](https://arxiv.org/abs/1506.05869)）及两层长短时记忆网络（[LSTM](http://people.idsia.ch/~juergen/lstm/)），同时使用了[seq2seq](https://www.tensorflow.org/tutorials/seq2seq)模型，其主要就是定义基本的LSTM结构作为循环体的基础结构，通过MultiRNNCell类实现深层循环神经网络，利用dropout策略在处理完的数据上运行tf.train操作，返回全部数据上的perplexity的值，具体实现参考实例代码[model]()&[train]()

## <a name="训练后数据"></a>训练后数据

* 数据加工完成后，Tensorflow默认会将训练结果保存为model.ckpt
* ErGo每次进行预测（即对话）时会加载相关的模型数据，返回接近最优的回答

#### 改进方法
在交互过程中可以看出不足的地方，下面列出可以提高ErGo性能的方法：
- 提供更多的数据库，以帮助从更大的会话语料库中学习
- LSTM结构参数的调优，如LSTM单元数、LSTM层数、优化器的选择、训练迭代次数


## <a name="如何变成自己的项目">如何变成自己的项目

1. 获取自己的数据副本，利用相关工具或脚本提取相关的对话序列生成自己的数据集
2. 利用word2vec为自己数据集中出现的单词生成单词向量。
3. 可以根据需求来拓展修改相关model函数
4. 训练模型
5. 参考[本地部署](https://github.com/cloudframeworks-tensorflow/user-guide-tensorflow#本地部署)执行数据训练及访问等

# <a name="更新计划"></a>更新计划

* `训练` 提供训练数据API接口
* `展示` 提供展示界面API接口
* `文档` 微信界面展示接入 

点击查看[历史更新](CHANGELOG.md)

# <a name="社群贡献"></a>社群贡献

+ QQ群: 621870673
+ [参与贡献](CONTRIBUTING.md)
+ [联系我们](mailto:info@goodrain.com)

-------

[云框架](ABOUT.md)系列主题，遵循[APACHE LICENSE 2.0](LICENSE.md)协议发布。

