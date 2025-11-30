# API 参考文档

本文档提供了 MyAutoGrad 框架的详细 API 参考。

## 目录

- [Variable 类](#variable-类)
- [数学运算](#数学运算)
- [激活函数](#激活函数)
- [损失函数](#损失函数)
- [张量操作](#张量操作)
- [计算图](#计算图)
- [优化器](#优化器)
- [循环神经网络](#循环神经网络)

## Variable 类

Variable 是框架的核心类，表示支持自动微分的变量。

### 构造函数

```cpp
// 标量构造函数
explicit Variable(double value, Nodetype type, const std::vector<size_t> &shape = {})

// 向量构造函数
explicit Variable(std::vector<double> &data, Nodetype type, const std::vector<size_t> &shape = {})

// 引用模式构造函数
explicit Variable(std::vector<VarPtr> original, std::vector<double*> &data, 
                std::vector<double*> &grad, Nodetype type=reference, 
                const std::vector<size_t> &shape = {})
```

### 节点类型

```cpp
enum Nodetype {
    intermediate,  // 中间节点
    parameter,     // 可训练参数
    input,         // 输入节点
    reference,     // 引用节点
};
```

### 主要方法

#### 数据访问

```cpp
// 获取数据
const DataView& data() const

// 获取梯度
const DataView& grad() const

// 获取标量值（仅适用于单元素）
double item() const

// 获取梯度标量值（仅适用于单元素）
double grad_item() const
```

#### 形状和维度

```cpp
// 获取形状
std::vector<size_t> shape() const

// 获取维度数
int ndim() const

// 获取元素总数
size_t size() const

// 检查是否为标量
bool is_scalar() const

// 检查是否为向量
bool is_vector() const

// 检查是否为矩阵
bool is_matrix() const
```

#### 计算方法

```cpp
// 前向计算
void calc()

// 反向传播
void backward(const std::vector<double> &grad_output = {})

// 清零梯度
void zero_grad()

// 递归清零梯度
void zero_grad_recursive()

// 使用梯度更新参数
void update(double learning_rate)
```

#### 索引访问

```cpp
// 通过多维索引获取元素
double& Item(const std::vector<int> &idx)
double& Item(size_t flat_index)

// 通过多维索引获取梯度
double& GradItem(const std::vector<int> &idx)
double& GradItem(size_t flat_index)

// 获取元素地址
double* ItemAddr(const std::vector<int> &idx)
double* ItemAddr(size_t flat_index)
```

## 数学运算

### 基础运算

```cpp
// 加法（支持广播）
VarPtr add(VarPtr a, VarPtr b, const std::string &name="")

// 减法（支持广播）
VarPtr sub(VarPtr a, VarPtr b, const std::string &name="")

// 元素级乘法（支持广播）
VarPtr mul_elementwise(VarPtr a, VarPtr b, const std::string &name="")

// 张量乘法（沿指定轴收缩）
VarPtr mul(VarPtr a, VarPtr b, int axis_a = -1, int axis_b = -1, const std::string &name = "")

// 元素级幂运算
VarPtr pow_elementwise(VarPtr a, double exponent, const std::string &name="")
```

### 聚合运算

```cpp
// 求和
VarPtr sum(VarPtr a, const std::string& name="")

// 多变量求和
VarPtr sum(std::vector<VarPtr> vars, const std::string& name="")

// 平均值
VarPtr mean(VarPtr a, const std::string& name="")
```

### 操作符重载

```cpp
VarPtr operator+(VarPtr a, VarPtr b)  // 加法
VarPtr operator-(VarPtr a, VarPtr b)  // 减法
VarPtr operator*(VarPtr a, VarPtr b)  // 乘法
VarPtr operator^(VarPtr a, double exponent)  // 幂运算
```

## 激活函数

```cpp
// ReLU 激活函数
VarPtr relu(VarPtr a, const std::string& name="")

// Sigmoid 激活函数
VarPtr sigmoid(VarPtr a, const std::string& name="")

// Tanh 激活函数
VarPtr tanh_activation(VarPtr a, const std::string& name="")
```

## 损失函数

```cpp
// 均方误差损失
VarPtr mse_loss(VarPtr predictions, VarPtr targets, const std::string& name="")

// 二元交叉熵损失
VarPtr binary_cross_entropy_loss(VarPtr predictions, VarPtr targets, const std::string& name="")
```

## 张量操作

### 切片操作

```cpp
// 基础切片
VarPtr slice(VarPtr input, const std::vector<int>& indices, const std::string& name="")

// 范围切片
VarPtr slice_indices(VarPtr input, const std::vector<idx_range>& indices, const std::string& name="")

// 字符串切片语法
VarPtr slice_(VarPtr input, const std::string slice_str, const std::string& name="")
```

### 形状变换

```cpp
// 展平张量
VarPtr flatten(VarPtr input, const std::string& name="")

// 拼接向量
VarPtr concat(VarPtr a, VarPtr b, const std::string& name="")

// 堆叠张量
VarPtr stack(std::vector<VarPtr> vars, const std::string& name="")
```

### 卷积和池化

```cpp
// 2D 卷积
VarPtr conv2d(VarPtr a, VarPtr b, const std::string& name="")

// 最大池化
VarPtr MaxPooling(VarPtr a, size_t filter_size=2, const std::string& name="")
```

### 线性层

```cpp
struct Linear {
    Linear(size_t input_dim, size_t output_dim, bool use_bias=true)
    VarPtr operator()(VarPtr input)
};
```

## 计算图

### ComputationGraph 类

```cpp
class ComputationGraph {
public:
    // 从输出节点构建计算图
    static ComputationGraph BuildFromOutput(const VarPtr& output_node)
    static ComputationGraph BuildFromOutput(const std::vector<VarPtr>& output_nodes)
    
    // 拓扑排序
    static std::vector<VarPtr> toposort(ComputationGraph &graph)
    
    // 根据名称获取节点
    VarPtr get_node_by_name(const string& name)
    
    // 保存/加载参数
    void SaveParams(string filename)
    void LoadParams(string filename)
    
    // 可视化计算图
    void Visualize(string filename)
    
    // 训练循环
    void fit(std::function<void(ComputationGraph* pgraph)> load_data, 
            int epochs, int n_samples, double learning_rate,
            std::function<void(ComputationGraph* pgraph)> print_info_before=nullptr,
            std::function<void(ComputationGraph* pgraph)> print_info_after=nullptr)
    
    // 打印图摘要
    void print_summary()
};
```

## 优化器

### Adam 优化器

```cpp
class AdamOptimizer {
public:
    AdamOptimizer(double learning_rate=0.001, double beta1=0.9, 
                 double beta2=0.999, double epsilon=1e-8)
    
    void set_parameter_nodes(std::vector<VarPtr> params)
    void update(VarPtr& var)
};
```

## 循环神经网络

### RecurrentOperation 类

```cpp
class RecurrentOperation {
public:
    RecurrentOperation(recurrent_op fn_, VarPtr hidden_input, VarPtr input)
    
    // 展开循环操作
    void expand(int times, bool multiple_outputs = false, 
               std::function<VarPtr(VarPtr)> output_transform = nullptr,
               std::function<VarPtr(VarPtr)> input_transform = nullptr)
};
```

### 预定义循环操作

```cpp
// 线性循环单元
recurrent_op linear_(bool use_bias=true)

// LSTM 单元
recurrent_op lstm_(size_t long_term_size, size_t short_term_size)
```

## 变量创建宏

为了方便变量创建和命名，框架提供了以下宏：

```cpp
#define MAKE_INPUT(name, ...) MAKE_VAR_(make_input, name, __VA_ARGS__)
#define MAKE_PARAM(name, ...) MAKE_VAR_(make_param, name, __VA_ARGS__)
#define MAKE_REF(name, ...) MAKE_VAR_(make_ref, name, __VA_ARGS__)
#define MAKE_VAR(name, ...) MAKE_VAR_(make_var, name, __VA_ARGS__)

#define ADD(name, ...) MAKE_VAR_(add, name, __VA_ARGS__)
#define SUB(name, ...) MAKE_VAR_(sub, name, __VA_ARGS__)
#define MUL(name, ...) MAKE_VAR_(mul, name, __VA_ARGS__)
// ... 更多操作宏
```

## 工具函数

### 数据创建

```cpp
// 随机向量
std::vector<double> vec_r(size_t size, double scale = 0.1)

// 零向量
std::vector<double> zero_vec(size_t size)
```

### 索引和切片

```cpp
// 创建范围
std::vector<int> range(int start, int stop, int step, int size)

// 解析切片字符串
std::vector<idx_range> parse_slices(const std::string slice_str, const std::vector<size_t>& shape)

// 广播索引计算
std::vector<int> get_broadcast_idx(const std::vector<int>& result_idx, const std::vector<size_t>& var_shape)
```

### 打印和调试

```cpp
// 打印向量
template<typename T>
void print_vec(std::ostream& os, const std::vector<T>& vec)

// 带格式的向量打印
template<typename T>
void print_vec(std::ostream& os, const std::vector<T>& vec, 
               std::string open_bracket, std::string delimiter, std::string close_bracket)