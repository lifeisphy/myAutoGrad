# 示例代码

MyAutoGrad 框架的实际应用示例，展示如何使用框架解决各种机器学习问题。

## 目录

1. [神经网络实验](#神经网络实验)
2. [高级应用示例](#高级应用示例)
3. [编译说明](#编译说明)

## 神经网络实验

### 实验 1: 多层感知机 (MLP)

使用多层感知机进行 MNIST 手写数字识别，展示全连接神经网络的训练和评估。

**代码文件**: [`experiment_mlp.cpp`](experiment_mlp.cpp)

**实验特点**:
- 784-128-10 网络结构
- ReLU 激活函数
- Adam 优化器
- MSE 损失函数
- 在线学习 (batch size = 1)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. experiment_mlp.cpp -o experiment_mlp && ./experiment_mlp
```

**预期结果**:
- 训练损失: ~0.0338
- 测试准确率: 100%
- 训练时间: ~45秒/epoch

### 实验 2: 卷积神经网络 (CNN)

使用卷积神经网络进行 MNIST 手写数字识别，展示卷积和池化操作的应用。

**代码文件**: [`experiment_cnn.cpp`](experiment_cnn.cpp)

**实验特点**:
- 2个卷积层 (32和48个3×3滤波器)
- 最大池化层
- 全连接层 (128个神经元)
- ReLU 激活函数
- Adam 优化器

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. experiment_cnn.cpp -o experiment_cnn && ./experiment_cnn
```

**预期结果**:
- 训练损失: ~0.0833
- 测试准确率: ~40%
- 训练时间: ~180秒/epoch
- 总参数量: 108,570

### 实验 3: 循环神经网络 (RNN)

使用循环神经网络进行正弦函数预测，展示序列建模和时间依赖学习。

**代码文件**: [`experiment_rnn.cpp`](experiment_rnn.cpp)

**实验特点**:
- 隐藏层大小: 32
- 输入维度: 10 (前10个值)
- 序列长度: 20个时间步
- 目标函数: sin(5x)
- 单值预测输出

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. experiment_rnn.cpp -o experiment_rnn && ./experiment_rnn
```

**预期结果**:
- 最终训练损失: ~0.000206
- 平均预测误差: ~0.0115
- 成功学习正弦函数模式

## 高级应用示例

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

### LSTM 文本生成

使用 LSTM 网络进行简单的字符级文本生成。

**代码文件**: [`lstm_text_generation.cpp`](lstm_text_generation.cpp)

**编译运行**:
```bash
conda activate gcc-15 && g++ -std=c++23 -I../.. lstm_text_generation.cpp -o lstm_text_generation && ./lstm_text_generation
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

这些示例涵盖了 MyAutoGrad 框架的各种应用场景，从基础的线性回归到复杂的深度学习模型。三个主要实验展示了框架在不同类型神经网络任务中的应用能力。