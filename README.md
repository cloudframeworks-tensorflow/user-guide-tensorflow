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
* [框架说明-业务](#框架说明-业务) 
* [框架说明-组件](#框架说明-组件)
    * [输入](#输入)
    * [处理](#处理)
        * [模型](#模型)
        * [训练](#训练)
    * [输出](#输出)
* [更新计划](#更新计划)
* [社群贡献](#社群贡献)

# <a name="快速部署"></a>快速部署

TODO @YSCING

# <a name="业务说明-业务"></a>框架说明-业务

基于TensorFlow自建聊天机器人ErGo可与用户完成智能对话，工作流程如下图所示：

<div align=center><img width="900" height="" src="./image/businessflow.png"/></div>

* 用户（User）发问经由ErGo进行分析
* ErGo根据分析在数据中检索（Retrieve）候选答案
* 整理候选答案并抽取（Extraction）作为回答的答案

# <a name="框架说明-组件"></a>框架说明-组件

<div align=center><img width="900" height="" src="./image/ergoarchitecture.png"/></div>

TODO @YSCING

## <a name="输入"></a>输入

## <a name="处理"></a>处理

### <a name="模型"></a>模型

### <a name="训练"></a>训练

### <a name="输出"></a>输出

# <a name="更新计划"></a>更新计划

* `文档` 
* `组件` 

点击查看[历史更新](CHANGELOG.md)

# <a name="社群贡献"></a>社群贡献

+ QQ群: 
+ [参与贡献](CONTRIBUTING.md)
+ [联系我们](mailto:info@goodrain.com)

-------

[云框架](ABOUT.md)系列主题，遵循[APACHE LICENSE 2.0](LICENSE.md)协议发布。

