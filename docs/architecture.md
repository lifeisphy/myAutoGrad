# 架构设计

MyAutoGrad 框架的内部架构和设计原理。

## 整体架构

MyAutoGrad 采用基于计算图的自动微分架构，主要由以下几个核心组件构成：

```
┌─────────────────────────────────────────────────────────────┐
│                    用户代码层                              │
├─────────────────────────────────────────────────────────────┤
│  Variable 类  │  数学运算  │  损失函数  │  张量操作      │
├─────────────────────────────────────────────────────────────┤
│                    计算图管理层                              │
├─────────────────────────────────────────────────────────────┤
│  内存管理  │  梯度计算  │  优化器  │  可视化工具        │
├─────────────────────────────────────────────────────────────┤
│                    底层存储层                               │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Variable 类

Variable 是框架的核心类，代表计算图中的节点。它包含以下关键属性：

```cpp
class Variable {
private:
    DataView data_;                    // 数据存储
    DataView grad_;                    // 梯度存储
    Nodetype type_;                   // 节点类型
    std::function<void(const DataView&)> grad_fn_;  // 梯度函数
    std::function<void()> forward_fn_;             // 前向函数
    std::vector<Edge*> children_;       // 子节点
    std::vector<Edge*> parents_;        // 父节点
    std::vector<size_t> shape_;         // 张量形状
    std::vector<VarPtr> ref;           // 引用变量列表
    std::string name;                   // 变量名称
    std::string operator_name;          // 操作名称
    bool updated_;                      // 更新状态
};
```

#### 节点类型

- **parameter**: 可训练参数，需要计算和存储梯度
- **input**: 输入数据，不需要梯度
- **intermediate**: 中间计算结果，需要梯度
- **reference**: 引用节点，共享其他节点的数据

#### 计算流程

1. **前向传播**: 通过 `calc()` 方法执行前向计算
2. **反向传播**: 通过 `backward()` 方法执行梯度计算
3. **梯度累积**: 通过 `accumulate_gradient()` 方法累积梯度

### 2. DataView 类

DataView 提供了灵活的数据存储和访问机制，支持两种模式：

```cpp
class DataView {
    std::vector<double*> dataview_;  // 指针模式（引用）
    std::vector<double> data_;        // 数据模式（拥有）
    bool references_;                 // 模式标志
};
```

#### 引用模式

- 存储指向其他数据的指针
- 用于实现切片、拼接等零拷贝操作
- 支持梯度传播到原始数据

#### 数据模式

- 拥有自己的数据副本
- 用于存储参数和中间结果
- 支持标准的读写操作

### 3. Edge 类

Edge 表示计算图中的边，连接父节点和子节点：

```cpp
struct Edge {
    VarPtr parent;        // 父节点
    VarPtr child;         // 子节点
    bool updated;         // 更新状态
    bool pass_grad;       // 是否传递梯度
};
```

#### 梯度控制

- `pass_grad` 控制是否向父节点传播梯度
- 支持实现梯度裁剪、梯度停止等高级功能

### 4. ComputationGraph 类

ComputationGraph 管理整个计算图的结构和执行：

```cpp
class ComputationGraph {
    std::vector<VarPtr> input_nodes;        // 输入节点
    std::vector<VarPtr> parameter_nodes;     // 参数节点
    std::vector<VarPtr> intermediate_nodes;  // 中间节点
    std::vector<VarPtr> reference_nodes;     // 引用节点
    std::vector<VarPtr> output_nodes;       // 输出节点
};
```

#### 图构建

通过 `BuildFromOutput()` 方法从输出节点反向构建计算图：

1. 从输出节点开始深度优先搜索
2. 根据节点类型分类存储
3. 建立完整的节点依赖关系

#### 拓扑排序

通过 `toposort()` 方法对节点进行拓扑排序：

1. 从输入节点和参数节点开始
2. 按依赖关系排序
3. 确保前向传播的正确顺序

## 自动微分机制

### 前向传播

前向传播按照拓扑顺序执行：

```cpp
void Variable::calc() {
    if (type_ != intermediate && type_ != reference) {
        updated_ = true;
        return;
    }
    
    // 递归计算子节点
    for (auto& edge : children_) {
        if (!edge->child->updated()) {
            edge->child->calc();
        }
    }
    
    // 执行当前节点的前向函数
    if (this->forward_fn_) {
        updated_ = true;
        this->forward_fn_();
    }
}
```

### 反向传播

反向传播采用累积梯度机制：

```cpp
void Variable::accumulate_gradient(const std::vector<double>& grad_input, bool accumulate = true) {
    if (!has_grad()) return;
    
    // 累积梯度
    if (accumulate) {
        for (size_t i = 0; i < grad_.size() && i < grad_input.size(); ++i) {
            if (!grad_.is_nullptr(i)) {
                grad_[i] += grad_input[i];
            }
        }
    }
    
    // 检查是否所有父节点都已发送梯度
    bool all_gradients_received = true;
    for (const auto& edge : parents()) {
        if (edge->pass_grad && edge->parent->has_grad()) {
            if (edge->updated == false) {
                all_gradients_received = false;
                break;
            }
        }
    }
    
    // 如果所有梯度都已接收，执行反向传播
    if (all_gradients_received) {
        if (grad_fn_) {
            grad_fn_(grad_);
        }
    }
}
```

### 梯度计算

每个操作都定义了对应的梯度函数。例如，加法操作的梯度计算：

```cpp
auto grad_fn = [result, result_size](const DataView &grad_output) {
    for (auto edge : result->children()) {
        auto node = edge->child;
        if (node == nullptr) continue;
        if (node->has_grad()) {
            std::vector<double> grad_node(node->size(), 0.0);
            for (size_t i = 0; i < result_size; i++) {
                std::vector<int> node_idx = get_broadcast_idx(
                    result->PlainItemIndex(i), node->shape());
                grad_node[node->ItemIndex(node_idx)] += grad_output[i];
            }
            edge->updated = true;
            node->accumulate_gradient(grad_node);
        }
    }
};
```

## 内存管理

### 智能指针

框架使用 `std::shared_ptr` 管理 Variable 对象的生命周期：

```cpp
using VarPtr = std::shared_ptr<class Variable>;
```

### 零拷贝操作

通过 DataView 的引用模式实现零拷贝操作：

1. **切片操作**: 直接引用原始数据，不创建副本
2. **拼接操作**: 组合多个数据引用
3. **展平操作**: 重新解释数据布局

### 内存优化

1. **梯度复用**: 中间节点的梯度在反向传播后可以释放
2. **延迟计算**: 只有在需要时才执行计算
3. **引用计数**: 自动管理内存释放

## 广播机制

### 广播规则

框架支持 NumPy 风格的广播机制：

1. 从右到左比较维度
2. 维度大小为 1 时可以广播
3. 缺失维度视为 1

### 广播实现

```cpp
std::vector<int> get_broadcast_idx(const std::vector<int>& result_idx, 
                                 const std::vector<size_t>& var_shape) {
    int result_dims = result_idx.size();
    if (var_shape.empty()) return {};
    
    std::vector<int> var_idx(var_shape.size());
    int offset = result_dims - var_shape.size();
    
    for (size_t j = 0; j < var_shape.size(); j++) {
        int idx_val = (var_shape[j] == 1) ? 0 : result_idx[j + offset];
        var_idx[j] = idx_val;
    }
    return var_idx;
}
```

## 优化器设计

### Adam 优化器

Adam 优化器实现了一阶和二阶矩估计：

```cpp
class AdamOptimizer {
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    std::map<VarPtr, std::pair<std::vector<double>, std::vector<double>>> moments_;
    
    void update(VarPtr& var) {
        auto& [m, v] = moments_[var];
        for (size_t i = 0; i < var->size(); ++i) {
            // 更新一阶矩
            m[i] = beta1_ * m[i] + (1 - beta1_) * var->grad()[i];
            // 更新二阶矩
            v[i] = beta2_ * v[i] + (1 - beta2_) * var->grad()[i] * var->grad()[i];
            // 偏差修正
            double m_hat = m[i] / (1 - beta1_);
            double v_hat = v[i] / (1 - beta2_);
            // 参数更新
            var->Item(i) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
};
```

## 循环神经网络支持

### RecurrentOperation 类

RecurrentOperation 提供了循环神经网络的抽象：

```cpp
class RecurrentOperation {
    VarPtr hidden_input;              // 初始隐藏状态
    VarPtr input;                     // 输入
    std::vector<VarPtr> params;       // 参数
    std::vector<VarPtr> hidden;      // 隐藏状态序列
    std::vector<VarPtr> inputs;      // 输入序列
    std::vector<VarPtr> outputs;     // 输出序列
    recurrent_op fn;                  // 循环操作函数
};
```

### 展开机制

通过 `expand()` 方法将循环操作展开为静态计算图：

1. 为每个时间步创建输入节点
2. 应用循环操作函数
3. 构建时间步之间的连接
4. 生成输出序列

### LSTM 实现

LSTM 通过组合多个线性层和激活函数实现：

```cpp
recurrent_op lstm_(size_t long_term_size, size_t short_term_size) {
    return [=](VarPtr hidden_state, VarPtr input, bool make_params, 
               std::vector<VarPtr>& params) -> VarPtr {
        // 分离隐藏状态和细胞状态
        auto h = slice_indices(hidden_state, {idx_range(0, long_term_size, 1)});
        auto c = slice_indices(hidden_state, {idx_range(long_term_size, hidden_dim, 1)});
        
        // 拼接输入和细胞状态
        auto combined = concat(c, input, "lstm_combined");
        
        // 创建或使用线性层
        // ... 线性层创建逻辑 ...
        
        // 计算门控
        auto f_t = sigmoid(lin1(combined), "lstm_f_t");  // 遗忘门
        auto i_t = sigmoid(lin2(combined), "lstm_i_t");  // 输入门
        auto g_t = tanh_activation(lin3(combined), "lstm_g_t");  // 候选值
        auto o_t = sigmoid(lin4(combined), "lstm_o_t");  // 输出门
        
        // 更新状态
        auto h_new = h * f_t + i_t * g_t;
        auto c_new = o_t * tanh_activation(h_new, "lstm_c_new");
        
        // 返回新的隐藏状态
        return concat(h_new, c_new, "lstm_hidden_output");
    };
}
```

## 可视化系统

### Graphviz 集成

计算图可视化通过生成 Graphviz DOT 格式实现：

```cpp
void ComputationGraph::Visualize(string filename) {
    std::ofstream ofs(filename);
    ofs << "digraph ComputationGraph {" << std::endl;
    
    // 生成节点定义
    for (auto& node : sorted) {
        ofs << "  \"" << node->name << "\" [label=<<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\">"<< std::endl;
        // 添加节点信息表格
        ofs << "\t</TABLE>\n\t> fillcolor="<< color << " style="<< "filled" << " ];" << std::endl;
    }
    
    // 生成边定义
    for (auto& node : sorted) {
        for (auto& edge : node->parents()) {
            if (edge->parent) {
                ofs << "  \"" << node->name << "\" -> \"" << edge->parent->name << "\"" << std::endl;
            }
        }
    }
    
    ofs << "}" << std::endl;
}
```

## 性能优化

### 计算优化

1. **延迟计算**: 只有在需要时才执行计算
2. **内存复用**: 通过引用模式减少内存分配
3. **批量操作**: 支持向量化计算

### 编译优化

1. **模板特化**: 针对不同大小的张量进行优化
2. **内联函数**: 减少函数调用开销
3. **编译器优化**: 利用编译器的优化能力

## 扩展机制

### 自定义操作

用户可以通过以下方式添加自定义操作：

```cpp
VarPtr custom_operation(VarPtr input, const std::string& name="") {
    // 创建输出变量
    auto result = make_var(std::vector<double>(input->size()), input->shape(), name);
    
    // 定义前向函数
    auto forward_fn = [input, result]() {
        // 实现前向计算
        for (size_t i = 0; i < input->size(); ++i) {
            result->Item(i) = /* 自定义计算 */;
        }
    };
    
    // 定义梯度函数
    auto grad_fn = [input, result](const DataView& grad_output) {
        if (input->has_grad()) {
            std::vector<double> grad_input(input->size());
            // 实现梯度计算
            for (size_t i = 0; i < input->size(); ++i) {
                grad_input[i] = /* 梯度计算 */;
            }
            input->accumulate_gradient(grad_input);
        }
    };
    
    // 设置函数并建立连接
    result->set_forward_fn(forward_fn);
    result->set_grad_fn(grad_fn);
    add_link(result, input);
    
    return result;
}
```

### Python 绑定

通过 cppyy 框架提供 Python 接口：

```python
import cppyy

# 加载 C++ 代码
cppyy.include("autograd.hpp")

# 使用 C++ 类
from cppyy.gbl import make_var, add, mul

# Python 代码中使用
x = make_var(2.0)
w = make_var(3.0)
y = add(mul(w, x), make_var(1.0))
```

这种架构设计使得 MyAutoGrad 既保持了高性能，又提供了灵活的扩展能力，能够满足各种深度学习应用的需求。