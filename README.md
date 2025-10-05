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

### 1. 基础标量运算

```cpp
#include "autograd.hpp"

int main() {
    // 创建变量，启用梯度计算
    auto x = make_var(2.0, true);
    auto y = make_var(3.0, true);
    
    // 计算 z = x^2 + 2*x*y + y^2
    auto x_squared = mul(x, x);
    auto y_squared = mul(y, y);
    auto xy = mul(x, y);
    auto two_xy = mul(make_var(2.0), xy);
    auto temp = add(x_squared, two_xy);
    auto z = add(temp, y_squared);
    
    // 反向传播计算梯度
    z->backward();
    
    std::cout << "z = " << z->item() << std::endl;
    std::cout << "∂z/∂x = " << x->grad_item() << std::endl;
    std::cout << "∂z/∂y = " << y->grad_item() << std::endl;
    
    return 0;
}
```

### 2. 激活函数

```cpp
auto x = make_var(-0.5, true);

auto relu_out = relu(x);
auto sigmoid_out = sigmoid(x);
auto tanh_out = tanh_activation(x);

// 反向传播会自动计算激活函数的导数
relu_out->backward();
```

### 3. 向量运算

```cpp
std::vector<double> data1 = {1.0, 2.0, 3.0};
std::vector<double> data2 = {4.0, 5.0, 6.0};

auto vec1 = make_var(data1, true);
auto vec2 = make_var(data2, true);

auto vec_sum = add(vec1, vec2);
auto total = sum(vec_sum);  // 求和所有元素

total->backward();  // 梯度会分发到每个元素
```

### 4. 损失函数

```cpp
std::vector<double> predictions = {0.8, 0.3, 0.9};
std::vector<double> targets = {1.0, 0.0, 1.0};

auto pred_var = make_var(predictions, true);
auto target_var = make_var(targets, false);

// 均方误差损失
auto mse = mse_loss(pred_var, target_var);

// 二元交叉熵损失
auto bce = binary_cross_entropy_loss(pred_var, target_var);
```

### 5. 简单神经网络训练

```cpp
// 网络参数
auto w = make_var(0.1, true);
auto b = make_var(0.0, true);

// 训练循环
for (int epoch = 0; epoch < 100; ++epoch) {
    w->zero_grad();
    b->zero_grad();
    
    // 前向传播
    auto y_pred = add(mul(w, x), b);
    auto loss = mse_loss(y_pred, y_target);
    
    // 反向传播
    loss->backward();
    
    // 参数更新
    double learning_rate = 0.01;
    auto new_w = w->data()[0] - learning_rate * w->grad()[0];
    auto new_b = b->data()[0] - learning_rate * b->grad()[0];
    
    w = make_var(new_w, true);
    b = make_var(new_b, true);
}
```

## API 参考

### Variable类

- `Variable(data, requires_grad)`: 构造函数
- `data()`: 获取数据
- `grad()`: 获取梯度
- `item()`: 获取标量值
- `backward()`: 反向传播
- `zero_grad()`: 清零梯度

### 基础运算

- `add(a, b)`: 加法
- `sub(a, b)`: 减法
- `mul(a, b)`: 乘法
- `div(a, b)`: 除法
- `pow(a, exponent)`: 幂运算
- `sum(a)`: 求和
- `mean(a)`: 平均值

### 激活函数

- `relu(x)`: ReLU激活函数
- `sigmoid(x)`: Sigmoid激活函数
- `tanh_activation(x)`: Tanh激活函数
- `leaky_relu(x, slope)`: Leaky ReLU激活函数

### 损失函数

- `mse_loss(predictions, targets)`: 均方误差损失
- `binary_cross_entropy_loss(predictions, targets)`: 二元交叉熵损失

## 技术细节

### 计算图构建

框架使用计算图来跟踪运算的依赖关系。每个Variable都保存了生成它的梯度函数和父节点的引用。

### 反向传播

反向传播通过递归调用梯度函数实现。梯度从输出节点开始向输入节点传播，使用链式法则计算每个参数的梯度。

### 内存管理

使用C++14的智能指针（`std::shared_ptr`）来自动管理内存，避免内存泄漏。

### 广播机制

支持不同形状张量之间的运算，实现了简单的广播机制（标量与向量运算）。

## 限制和扩展

### 当前限制

- 仅支持标量和一维向量
- 没有实现高维张量运算
- 没有GPU加速支持
- 梯度计算不支持高阶导数

### 可能的扩展

- 添加多维张量支持
- 实现更多激活函数和损失函数
- 添加优化器类（SGD、Adam等）
- 实现卷积和池化操作
- 添加模型序列化功能

## 许可证

这个项目是一个教学示例，可以自由使用和修改。