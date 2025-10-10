Note: this demo is based on AI generated code with modification. Use with caution.

# C++ 基础自动微分框架

这是一个用C++实现的基础自动微分框架，支持标量和向量的自动微分计算。该框架实现了反向传播算法，可以自动计算复杂函数的梯度。

## 主要特性

- **Variable类**: 支持自动微分的核心数据结构
- **基础数学运算**: 加法、减法、乘法、除法、幂运算等
- **激活函数**: ReLU、Sigmoid、Tanh、Leaky ReLU
- **损失函数**: 均方误差(MSE)、二元交叉熵(BCE)
- **向量运算**: 支持向量运算和广播
- **内存管理**: 使用智能指针自动管理内存

## 文件结构

```
├── autograd.hpp        # 自动微分框架头文件
├── 1.cpp              # 主演示程序
├── test_autograd.cpp  # 详细测试程序
├── Makefile           # 编译脚本
└── README.md          # 说明文档
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
g++ -std=c++14 -Wall -Wextra -O2 -o autograd_demo 1.cpp

# 编译测试程序
g++ -std=c++14 -Wall -Wextra -O2 -o autograd_test test_autograd.cpp
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
```

## 使用示例
See [testcases](./testcases/) for more details