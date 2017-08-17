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
    * [输入](#输入)
    * [处理](#处理)
        * [模型](#模型)
        * [训练](#训练)
    * [输出](#输出)
* [更新计划](#更新计划)
* [社群贡献](#社群贡献)

# <a name="快速部署"></a>快速部署

## 一键部署

[一键部署在好雨云平台](http://app.goodrain.com/group/detail/30/)

## 本地部署

1. [准备Docker环境]()
 
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

# <a name="技术流程"></a>技术流程

ErGo的技术流程整体可分为**输入**、**处理**（包括模型及训练）、**输出**三阶段，如下图所示：

<div align=center><img width="900" height="" src="./image/ergo-flow.png"/></div>

* 在输入阶段，ErGo加载Data（语料）并进行数据处理
* 处理完成后，由训练模型（Training Model）加载并进行反复训练
* 完成训练后，ErGo即可根据训练好的数据进行相关预测，即与用户完成对话

## <a name="输入"></a>输入

输入阶段本例选用语料来源于[Cornell_Movie-Dialogs_Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## <a name="处理"></a>处理

Tensorflow支持多种数据读取方式.默认使用从文件中读取，标准编码、解码器。

### <a name="模型"></a>模型

选择的是循环神经网络(RNN)及其的一个重要结构长短时记忆网络(LSTM).

- seq2seq model
- 2 LTSM layers

定义基本的LSTM结构作为循环体的基础结构，默认lstm_hidden_size默认为512  
在通过MultiRNNCell类实现深层循环神经网络了，同时使用dropout  
具体参考 代码的model部分 

参考 [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)

### <a name="训练"></a>训练

使用给定的模型model在数据data上运行train并返回全部数据上的perplexity的值。
具体实现参考代码main.py训练部分.
其中lr和dropout值的设定对训练结果的影响最大.  

## <a name="输出"></a>输出

默认会将训练结果保存为model.ckpt.

## <a name="如何变成自己的项目">如何变成自己的项目

* 替换训练数据
* 替换展示界面

理论上只需提供自己项目相关的训练数据即可，后期会支持相关api接口调用。

# <a name="更新计划"></a>更新计划

* `训练` 支持云平台训练 
* `展示界面` 提供API接口
* `展示界面` 通过微信展示 

点击查看[历史更新](CHANGELOG.md)

# <a name="社群贡献"></a>社群贡献

+ QQ群: 621870673
+ [参与贡献](CONTRIBUTING.md)
+ [联系我们](mailto:info@goodrain.com)

-------

[云框架](ABOUT.md)系列主题，遵循[APACHE LICENSE 2.0](LICENSE.md)协议发布。

