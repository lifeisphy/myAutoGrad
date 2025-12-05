# MyAutoGrad - C++ 自动微分框架

MyAutoGrad 是一个用 C++ 实现的高性能自动微分框架，支持构建和训练深度神经网络。该框架实现了完整的反向传播算法，支持标量、向量和多维张量的自动微分计算。

## 主要特性

### 核心功能

- **Variable 类**: 支持自动微分的核心数据结构
- **计算图管理**: 自动构建和管理计算图
- **反向传播**: 高效的梯度计算和传播
- **内存管理**: 使用智能指针和 DataView 实现高效的内存管理

### 支持的操作

- **基础数学运算**: 加法、减法、乘法、除法、幂运算等
- **激活函数**: ReLU、Sigmoid、Tanh、Leaky ReLU
- **损失函数**: 均方误差(MSE)、二元交叉熵(BCE)
- **张量操作**: 卷积、池化、切片、拼接、展平等
- **向量运算**: 支持向量运算和广播

### 高级特性

- **循环神经网络**: 支持 RNN 和 LSTM 结构
- **优化器**: Adam 优化器
- **可视化**: 计算图可视化功能
- **参数保存/加载**: 模型参数的持久化
- **Python 绑定**: 通过 cppyy 支持 Python 调用

## 项目结构

```{plaintext}
├── autograd.hpp        # 主框架头文件
├── variable.hpp        # Variable 类定义
├── operations.hpp      # 数学运算实现
├── graph.hpp          # 计算图管理
├── optimizer.hpp      # 优化器实现
├── dataview.hpp       # 数据视图类
├── utils.hpp          # 工具函数
├── recurrent.hpp      # 循环神经网络支持
├── Makefile           # 编译脚本
├── requirements.txt   # Python 依赖
├── test/              # 测试和示例
│   ├── demo.cpp       # 基础演示
│   ├── test.cpp       # 单元测试
│   └── ...            # 其他测试文件
├── docs/              # 文档目录
│   ├── overview.md    # 本文档
│   ├── api/           # API 文档
│   ├── examples/      # 示例代码
│   └── architecture.md # 架构设计
```

## 快速开始

### 编译要求

- C++23 兼容的编译器 (推荐 GCC 13+ 或 Clang 16+)
- Make 工具
- Python 3.8+ (可选，用于 Python 绑定)

### 编译和运行

```bash
# 编译所有程序
make all

# 运行基础演示
make run

# 运行测试程序
make test

# 清理生成的文件
make clean
```

### 简单示例

```cpp
#include "autograd.hpp"

int main() {
    // 创建变量
    auto x = make_param(2.0);
    auto w = make_param(3.0);
    auto b = make_param(1.0);
    
    // 构建计算图: y = w * x + b
    auto y = add(mul(w, x), b);
    
    // 前向计算
    y->calc();
    std::cout << "y = " << y->item() << std::endl;  // 输出: y = 7
    
    // 反向传播
    y->backward();
    std::cout << "dw = " << w->grad_item() << std::endl;  // 输出: dw = 2
    std::cout << "dx = " << x->grad_item() << std::endl;  // 输出: dx = 3
    
    return 0;
}
```

## 文档导航

- [API 参考](api/README.md) - 详细的 API 文档
- [示例](examples/README.md) - 实际应用示例
- [架构设计](architecture.md) - 框架内部设计说明
