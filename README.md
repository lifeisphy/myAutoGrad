Note: this demo is based on AI generated code with modification. Use with caution.

# C++ 基础自动微分框架

这是一个用C++实现的基础自动微分框架，支持标量和向量的自动微分计算。该框架实现了反向传播算法，可以自动计算复杂函数的梯度。现已支持**CNN (卷积神经网络)** 架构！

## 主要特性

- **Variable类**: 支持自动微分的核心数据结构
- **基础数学运算**: 加法、减法、乘法、除法、幂运算等
- **激活函数**: ReLU、Sigmoid、Tanh、Leaky ReLU
- **损失函数**: 均方误差(MSE)、二元交叉熵(BCE)、Softmax交叉熵
- **向量运算**: 支持向量运算和广播
- **CNN操作**: 卷积(Conv2D)、池化(MaxPool2D)、展平(Flatten)、形状变换(Reshape)
- **内存管理**: 使用智能指针自动管理内存

## 文件结构

```
├── autograd.hpp              # 自动微分框架头文件
├── Makefile                  # 编译脚本
├── README.md                 # 说明文档
├── docs/
│   └── CNN_ARCHITECTURE.md   # CNN架构详细文档
└── test/
    ├── demo.cpp              # 主演示程序
    ├── test.cpp              # 测试程序
    ├── mnist_cnn_demo.cpp    # CNN MNIST演示程序
    ├── mnist_cnn.py          # Python CNN示例
    └── test_linear.py        # 线性回归测试
```

## 编译和运行

### 编译

使用Makefile编译所有程序：

```bash
make all
```

或者分别编译：

```bash
# 编译主演示程序
g++ -std=c++23 -Wall -Wextra -O2 -o autograd_demo test/demo.cpp

# 编译测试程序
g++ -std=c++23 -Wall -Wextra -O2 -o autograd_test test/test.cpp

# 编译CNN演示程序
g++ -std=c++23 -Wall -Wextra -O2 -o test/mnist_cnn_demo test/mnist_cnn_demo.cpp
```

### 运行

```bash
# 运行主演示程序
make run
# 或者
./autograd_demo

# 运行测试程序
make test
# 或者
./autograd_test

# 运行CNN演示程序
make cnn
# 或者
./test/mnist_cnn_demo
```

## CNN for MNIST

本框架现在支持完整的CNN架构用于MNIST手写数字识别！

### 架构概览

```
Input (28×28×1)
    ↓
Conv2D (1→8, 3×3) → ReLU → MaxPool2D (2×2)
    ↓
Conv2D (8→16, 3×3) → ReLU → MaxPool2D (2×2)
    ↓
Flatten (400) → Dense (400→128) → ReLU
    ↓
Dense (128→10) → Softmax
```

### 快速开始

```bash
# 编译并运行CNN演示
make cnn

# 查看详细文档
cat docs/CNN_ARCHITECTURE.md
```

**输出示例：**
```
Epoch 1, Sample 1 (digit=0), Loss: 2.3272
Epoch 1, Sample 5 (digit=4), Loss: 2.2664
Epoch 1 - Average Loss: 2.2985
...
✓ CNN architecture successfully built and tested
```

### 主要特性

- **Conv2D**: 2D卷积层，支持自定义通道数、卷积核大小、步长和填充
- **MaxPool2D**: 最大池化层，支持自定义池化窗口和步长
- **Flatten**: 将多维张量展平为一维
- **Reshape**: 改变张量形状
- **Softmax Cross Entropy**: 分类任务的损失函数

### 参数数量

- Conv1: 80 参数
- Conv2: 1,168 参数
- FC1: 51,328 参数
- FC2: 1,290 参数
- **总计: 53,866 参数**

### 详细文档

完整的CNN架构说明、使用示例和训练技巧请参考：[docs/CNN_ARCHITECTURE.md](docs/CNN_ARCHITECTURE.md)

## 使用示例

### 基础示例
See [test/demo.cpp](test/demo.cpp) for more details

### CNN示例
See [test/mnist_cnn_demo.cpp](test/mnist_cnn_demo.cpp) for complete CNN implementation