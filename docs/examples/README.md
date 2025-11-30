# 示例代码

本文档提供了 MyAutoGrad 框架的实际应用示例，展示了如何使用框架解决各种机器学习问题。

## 目录

1. [基础示例](#基础示例)
2. [神经网络示例](#神经网络示例)
3. [计算机视觉示例](#计算机视觉示例)
4. [自然语言处理示例](#自然语言处理示例)
5. [高级应用示例](#高级应用示例)

## 基础示例

### 线性回归

演示如何使用 MyAutoGrad 实现简单的线性回归模型。

**代码文件**: [`linear_regression.cpp`](linear_regression.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. linear_regression.cpp -o linear_regression && ./linear_regression
```

### 逻辑回归

演示二分类问题的逻辑回归实现。

**代码文件**: [`logistic_regression.cpp`](logistic_regression.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. logistic_regression.cpp -o logistic_regression && ./logistic_regression
```

## 神经网络示例

### 多层感知机

使用多层感知机解决 XOR 问题，展示非线性分类能力。

**代码文件**: [`multilayer_perceptron.cpp`](multilayer_perceptron.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. multilayer_perceptron.cpp -o multilayer_perceptron && ./multilayer_perceptron
```

### 使用 Adam 优化器

演示如何使用 Adam 优化器进行模型训练。

**代码文件**: [`adam_optimizer.cpp`](adam_optimizer.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. adam_optimizer.cpp -o adam_optimizer && ./adam_optimizer
```

### 高级功能示例

演示框架的高级功能，包括自定义操作和复杂网络结构。

**代码文件**: [`advanced_features.cpp`](advanced_features.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. advanced_features.cpp -o advanced_features && ./advanced_features
```

### CNN 网络

演示卷积神经网络的实现，包括卷积层和池化操作。

**代码文件**: [`cnn_network.cpp`](cnn_network.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. cnn_network.cpp -o cnn_network && ./cnn_network
```

### 模型训练

演示完整的模型训练流程，包括数据加载、训练循环和评估。

**代码文件**: [`model_training.cpp`](model_training.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. model_training.cpp -o model_training && ./model_training
```

## 计算机视觉示例

### 简单的 CNN

演示卷积神经网络的基本操作，包括卷积和池化。

**代码文件**: [`simple_cnn.cpp`](simple_cnn.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. simple_cnn.cpp -o simple_cnn && ./simple_cnn
```

### MNIST 数字识别（简化版）

使用神经网络进行手写数字识别的简化示例。

**代码文件**: [`mnist_classification.cpp`](mnist_classification.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. mnist_classification.cpp -o mnist_classification && ./mnist_classification
```

## 自然语言处理示例

### 简单的 RNN

演示循环神经网络的基本实现和训练过程。

**代码文件**: [`simple_rnn.cpp`](simple_rnn.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. simple_rnn.cpp -o simple_rnn && ./simple_rnn
```

### LSTM 文本生成

使用 LSTM 网络进行简单的字符级文本生成。

**代码文件**: [`lstm_text_generation.cpp`](lstm_text_generation.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. lstm_text_generation.cpp -o lstm_text_generation && ./lstm_text_generation
```

## 高级应用示例

### 计算图可视化

演示如何可视化计算图结构，便于调试和理解模型。

**代码文件**: [`graph_visualization.cpp`](graph_visualization.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. graph_visualization.cpp -o graph_visualization && ./graph_visualization
```

### 模型保存和加载

演示如何保存和加载模型参数，实现模型的持久化。

**代码文件**: [`model_save_load.cpp`](model_save_load.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. model_save_load.cpp -o model_save_load && ./model_save_load
```

### 使用宏简化代码

展示如何使用框架提供的宏来简化代码编写。

**代码文件**: [`macro_usage.cpp`](macro_usage.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. macro_usage.cpp -o macro_usage && ./macro_usage
```

## 编译说明

所有示例都可以使用以下通用命令编译：

```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. [filename].cpp -o [output_name] && ./[output_name]
```

或者使用项目提供的 Makefile：

```bash
conda activate gcc-15 && make [example_name]
```

这些示例涵盖了 MyAutoGrad 框架的各种应用场景，从基础的线性回归到复杂的深度学习模型。通过这些示例，您可以学习如何构建自己的机器学习应用。