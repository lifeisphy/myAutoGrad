# MyAutoGrad - C++ 自动微分框架

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B23)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen.svg)](docs/)

一个用 C++ 实现的高性能自动微分框架，支持构建和训练深度神经网络。该框架实现了完整的反向传播算法，支持标量、向量和多维张量的自动微分计算。

> **注意**: 本项目基于 AI 生成的代码并进行修改，请谨慎使用。

## ✨ 主要特性

### 🧠 核心功能

- **Variable 类**: 支持自动微分的核心数据结构
- **计算图管理**: 自动构建和管理计算图
- **反向传播**: 高效的梯度计算和传播
- **内存管理**: 使用智能指针和 DataView 实现高效的内存管理

### 🔧 支持的操作

- **基础数学运算**: 加法、减法、乘法、除法、幂运算等
- **激活函数**: ReLU、Sigmoid、Tanh、Leaky ReLU
- **损失函数**: 均方误差(MSE)、二元交叉熵(BCE)
- **张量操作**: 卷积、池化、切片、拼接、展平等
- **向量运算**: 支持向量运算和广播

### 🚀 高级特性

- **循环神经网络**: 支持 RNN 和 LSTM 结构
- **优化器**: Adam 优化器
- **可视化**: 计算图可视化功能
- **参数保存/加载**: 模型参数的持久化
- **Python 绑定**: 通过 cppyy 支持 Python 调用

## 📁 项目结构

```{text}
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
│   ├── mnist.cpp      # MNIST 手写数字识别
│   └── ...            # 其他测试文件
└── docs/              # 文档目录
    ├── overview.md    # 概述文档
    ├── api/           # API 文档
    ├── examples/      # 示例代码
    └── architecture.md # 架构设计
```

## 🚀 快速开始

### 编译要求

- C++23 兼容的编译器 (推荐 GCC 13+ 或 Clang 16+)
- Make 工具
- Python 3.8+ (可选，用于 Python 绑定)

### 安装依赖

```bash
# 安装 Python 依赖（可选）
pip install -r requirements.txt
```

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

# 查看帮助
make help
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

## 📚 文档

- [📖 概述](docs/overview.md) - 框架介绍和快速开始
- [🔧 API 参考](docs/api/README.md) - 详细的 API 文档
- [💡 示例](docs/examples/README.md) - 实际应用示例
- [🏗️ 架构设计](docs/architecture.md) - 框架内部设计说明

## 🎯 使用示例

### 线性回归

```cpp
// 创建变量
auto x = make_input(0.0);
auto w = make_param(0.1);
auto b = make_param(0.1);
auto target = make_input(0.0);

// 构建模型
auto y_pred = add(mul(w, x), b);
auto loss = mse_loss(y_pred, target);

// 训练循环
for (int epoch = 0; epoch < 100; ++epoch) {
    x->set_input(training_data[epoch]);
    target->set_input(labels[epoch]);
    
    loss->zero_grad_recursive();
    loss->calc();
    loss->backward();
    
    w->update(learning_rate);
    b->update(learning_rate);
}
```

### 神经网络

```cpp
// 多层感知机
auto W1 = make_param(vec_r(input_size * hidden_size), {hidden_size, input_size});
auto b1 = make_param(vec_r(hidden_size), {hidden_size});
auto W2 = make_param(vec_r(hidden_size * output_size), {output_size, hidden_size});
auto b2 = make_param(vec_r(output_size), {output_size});

// 前向传播
auto z1 = add(mul(W1, x, 0, 0), b1);
auto a1 = relu(z1);
auto z2 = add(mul(W2, a1, 0, 0), b2);
auto output = z2;
```

### 卷积神经网络

```cpp
// 卷积层
auto conv_weights = make_param(vec_r(3 * 3 * 32), {3, 3, 32});
auto conv_out = conv2d(input, conv_weights);
auto relu_out = relu(conv_out);
auto pool_out = MaxPooling(relu_out, 2);
```

### 循环神经网络

```cpp
// LSTM
auto lstm_op = lstm_(hidden_size, hidden_size);
auto lstm = RecurrentOperation(lstm_op, hidden_state, input);
lstm.expand(seq_length);
auto outputs = lstm.outputs;
```

## 🔄 最近更新

根据 git 记录，最近的重大更新包括：

- **🔄 RNN 网络支持**: 添加了循环神经网络和 LSTM 支持
- **📊 可视化功能**: 新增计算图可视化功能
- **⚡ Adam 优化器**: 实现了 Adam 优化算法
- **💾 参数保存/加载**: 支持模型参数的持久化
- **🐍 Python 绑定**: 通过 cppyy 提供 Python 接口
- **🔧 高级张量操作**: 卷积、池化、切片等操作

## 🧪 测试

项目包含全面的测试套件：

```bash
# 运行基础测试
./autograd_test

# 运行 MNIST 示例
./mnist train

# 运行演示程序
./autograd_demo
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

### 开发指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 📖 引用

如果您在研究或项目中使用了 MyAutoGrad，请考虑引用本仓库：

```bibtex
@software{myautograd,
  title={MyAutoGrad: C++ Automatic Differentiation Framework},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/myAutoGrad}
}
```

## 🙏 致谢

- 感谢所有贡献者的支持
- 灵感来源于 PyTorch、TensorFlow 等优秀框架
- 特别感谢开源社区的支持

## 📞 联系方式

- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/myAutoGrad/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/myAutoGrad/discussions)

---

⭐ 如果这个项目对您有帮助，请给我们一个 Star！
