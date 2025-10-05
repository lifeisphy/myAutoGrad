/**
 * 基础自动微分框架 (Basic Autograd Framework) - C++版本
 * 支持标量和向量的自动微分计算
 */

#include <iostream>
#include <vector>
#include <memory>
#include <functional>
#include <cmath>
#include <iomanip>

// 前向声明
class Variable;
using VarPtr = std::shared_ptr<Variable>;

/**
 * 支持自动微分的变量类
 */
class Variable {
private:
    std::vector<double> data_;
    std::vector<double> grad_;
    bool requires_grad_;
    std::function<void(const std::vector<double>&)> grad_fn_;
    std::vector<VarPtr> children_;
    std::vector<size_t> shape_;
public:
    /**
     * 构造函数
     * @param data 数值数据
     * @param requires_grad 是否需要梯度计算
     */
    explicit Variable(double value, bool requires_grad = false, const std::vector<size_t>& shape = {}) 
        : data_({value}), requires_grad_(requires_grad), shape_(shape) {
        if (requires_grad_) {
            grad_ = std::vector<double>(1, 0.0);
        }
        if (shape_.empty()) {
            shape_ = { data_.size()};
        }
        check_validity();
    }
    
    /**
     * 向量构造函数
     * @param data 数值向量
     * @param requires_grad 是否需要梯度计算
     */
    explicit Variable(const std::vector<double>& data, bool requires_grad = false, const std::vector<size_t>& shape = {})
        : data_(data), requires_grad_(requires_grad), shape_(shape) {
        if (requires_grad_) {
            grad_ = std::vector<double>(data.size(), 0.0);
        }
        if (shape_.empty()) {
            shape_ = { data_.size()};
        }
        check_validity();
    }
    void check_validity(){
        size_t total = 1;
        for(auto dim: shape_){
            if( dim <= 0){
                throw std::runtime_error("Invalid shape dimension");
            }
            total *= dim;
        }
        if(total != data_.size()){
            throw std::runtime_error("Data size does not match shape");
        }
    }
    // check if it is some simple shapes like scalar, vector and matrix
    bool is_scalar() const {
        return size() == 1;
    }
    bool is_vector() const {
        return shape_.size() == 1 && shape_[0] > 1;
    }
    bool is_matrix() const { 
        return shape_.size() == 2 && shape_[0] > 1 && shape_[1] > 1;
    }
    // 访问器
    const std::vector<double>& data() const { return data_; }
    const std::vector<double>& grad() const { return grad_; }
    bool requires_grad() const { return requires_grad_; }
    size_t size() const { return data_.size(); }
    std::vector<size_t> shape() const { return shape_; }
    // update the data using gradient and learning rate
    void update(double learning_rate) {
        for(size_t i= 0 ;i < data_.size(); i++){
            data_[i] -= learning_rate * grad_[i];
        }
    }
    // 获取标量值（仅适用于单元素）
    double item() const {
        if (!is_scalar()) {
            throw std::runtime_error("item() can only be called on scalar variables");
        }
        return data_[0];
    }
   
    // 获取梯度标量值（仅适用于单元素）
    double grad_item() const {
        if (!requires_grad_ || !is_scalar()) {
            throw std::runtime_error("grad_item() can only be called on scalar variables with gradients");
        }
        return grad_[0];
    }
    
    /**
     * 设置梯度函数
     */
    void set_grad_fn(std::function<void(const std::vector<double>&)> grad_fn) {
        grad_fn_ = grad_fn;
    }
    
    /**
     * 添加子节点
     */
    void add_child(VarPtr child) {
        children_.push_back(child);
    }
    
    /**
     * 反向传播计算梯度
     * @param grad_output 从上游传来的梯度
     */
    void backward(const std::vector<double>& grad_output = {}) {
        if (!requires_grad_) return;
        
        std::vector<double> grad_out = grad_output;
        
        // 如果是标量输出且没有指定梯度，设为1 (优化目标)
        if (grad_out.empty()) {
            grad_out = std::vector<double>(data_.size(), 1.0);
        }
        
        // 累积梯度
        if (grad_.size() != grad_out.size()) {
            grad_.resize(grad_out.size(), 0.0);
        }
        
        for (size_t i = 0; i < grad_.size(); ++i) {
            grad_[i] += grad_out[i];
        }
        
        // 如果有梯度函数，继续反向传播
        if (grad_fn_) {
            grad_fn_(grad_out);
        }
    }
    
    /**
     * 清零梯度
     */
    void zero_grad() {
        if (requires_grad_) {
            std::fill(grad_.begin(), grad_.end(), 0.0);
        }
    }
    
    /**
     * 打印变量信息
     */
    void print() const {
        std::cout << "Variable(data=[";
        for (size_t i = 0; i < data_.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << data_[i];
            if (i < data_.size() - 1) std::cout << ", ";
        }
        std::cout << "], requires_grad=" << std::boolalpha << requires_grad_;
        
        if (requires_grad_ && !grad_.empty()) {
            std::cout << ", grad=[";
            for (size_t i = 0; i < grad_.size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << grad_[i];
                if (i < grad_.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << ")" << std::endl;
    }
};

// 工具函数：创建Variable的智能指针
VarPtr make_var(double value, bool requires_grad = false) {
    return std::make_shared<Variable>(value, requires_grad);
}

VarPtr make_var(const std::vector<double>& data, bool requires_grad = false, const std::vector<size_t>& shape = {}) {
    return std::make_shared<Variable>(data, requires_grad, shape);
}

/**
 * 基础数学运算函数
 */

// 加法运算
VarPtr add(VarPtr a, VarPtr b) {
    if (a->size() != b->size() && a->size() != 1 && b->size() != 1) {
        throw std::runtime_error("Incompatible sizes for addition");
    }
    
    size_t result_size = std::max(a->size(), b->size());
    std::vector<double> result_data(result_size);
    
    // 执行加法（支持广播）
    for (size_t i = 0; i < result_size; ++i) {
        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
        result_data[i] = a_val + b_val;
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, b, result_size](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size(), 0.0);
                if (a->size() == 1) {
                    // partial theta / partial a = sum_i (partial theta / partial result_i) * (partial result_i / partial a)
                    // = sum_i grad_output[i] * 1
                    // 标量广播：求和所有梯度
                    for (double grad : grad_output) {
                        grad_a[0] += grad;
                    }
                } else {
                    grad_a = grad_output;
                }
                a->backward(grad_a);
            }
            
            if (b->requires_grad()) {
                std::vector<double> grad_b(b->size(), 0.0);
                if (b->size() == 1) {
                    // 标量广播：求和所有梯度
                    for (double grad : grad_output) {
                        grad_b[0] += grad;
                    }
                } else {
                    grad_b = grad_output;
                }
                b->backward(grad_b);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
        result->add_child(b);
    }
    
    return result;
}

// 减法运算
VarPtr sub(VarPtr a, VarPtr b) {
    if (a->size() != b->size() && a->size() != 1 && b->size() != 1) {
        throw std::runtime_error("Incompatible sizes for subtraction");
    }
    
    size_t result_size = std::max(a->size(), b->size());
    std::vector<double> result_data(result_size);
    
    for (size_t i = 0; i < result_size; ++i) {
        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
        result_data[i] = a_val - b_val;
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, b](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size(), 0.0);
                if (a->size() == 1) {
                    for (double grad : grad_output) {
                        grad_a[0] += grad;
                    }
                } else {
                    grad_a = grad_output;
                }
                a->backward(grad_a);
            }
            
            if (b->requires_grad()) {
                std::vector<double> grad_b(b->size(), 0.0);
                if (b->size() == 1) {
                    for (double grad : grad_output) {
                        grad_b[0] -= grad;  // 减法的梯度是负的
                    }
                } else {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        grad_b[i] = -grad_output[i];
                    }
                }
                b->backward(grad_b);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
        result->add_child(b);
    }
    
    return result;
}

// 乘法运算
VarPtr mul(VarPtr a, VarPtr b) {
    if (a->size() != b->size() && a->size() != 1 && b->size() != 1) {
        throw std::runtime_error("Incompatible sizes for multiplication");
    }
    
    size_t result_size = std::max(a->size(), b->size());
    std::vector<double> result_data(result_size);
    
    for (size_t i = 0; i < result_size; ++i) {
        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
        result_data[i] = a_val * b_val;
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, b](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size(), 0.0);
                if (a->size() == 1) {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
                        grad_a[0] += grad_output[i] * b_val;
                    }
                } else {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
                        grad_a[i] = grad_output[i] * b_val;
                    }
                }
                a->backward(grad_a);
            }
            
            if (b->requires_grad()) {
                std::vector<double> grad_b(b->size(), 0.0);
                if (b->size() == 1) {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
                        grad_b[0] += grad_output[i] * a_val;
                    }
                } else {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
                        grad_b[i] = grad_output[i] * a_val;
                    }
                }
                b->backward(grad_b);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
        result->add_child(b);
    }
    
    return result;
}

// 除法运算
VarPtr div(VarPtr a, VarPtr b) {
    if (a->size() != b->size() && a->size() != 1 && b->size() != 1) {
        throw std::runtime_error("Incompatible sizes for division");
    }
    
    size_t result_size = std::max(a->size(), b->size());
    std::vector<double> result_data(result_size);
    
    for (size_t i = 0; i < result_size; ++i) {
        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
        if (std::abs(b_val) < 1e-10) {
            throw std::runtime_error("Division by zero");
        }
        result_data[i] = a_val / b_val;
    }
    
    bool requires_grad = a->requires_grad() || b->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, b](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size(), 0.0);
                if (a->size() == 1) {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
                        grad_a[0] += grad_output[i] / b_val;
                    }
                } else {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
                        grad_a[i] = grad_output[i] / b_val;
                    }
                }
                a->backward(grad_a);
            }
            
            if (b->requires_grad()) {
                std::vector<double> grad_b(b->size(), 0.0);
                if (b->size() == 1) {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
                        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
                        grad_b[0] += -grad_output[i] * a_val / (b_val * b_val);
                    }
                } else {
                    for (size_t i = 0; i < grad_output.size(); ++i) {
                        double a_val = (a->size() == 1) ? a->data()[0] : a->data()[i];
                        double b_val = (b->size() == 1) ? b->data()[0] : b->data()[i];
                        grad_b[i] = -grad_output[i] * a_val / (b_val * b_val);
                    }
                }
                b->backward(grad_b);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
        result->add_child(b);
    }
    
    return result;
}

// 幂运算
VarPtr pow(VarPtr a, double exponent) {
    std::vector<double> result_data(a->size());
    
    for (size_t i = 0; i < a->size(); ++i) {
        result_data[i] = std::pow(a->data()[i], exponent);
    }
    
    bool requires_grad = a->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, exponent](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size());
                for (size_t i = 0; i < a->size(); ++i) {
                    // d/dx (x^n) = n * x^(n-1)
                    grad_a[i] = grad_output[i] * exponent * std::pow(a->data()[i], exponent - 1);
                }
                a->backward(grad_a);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
    }
    
    return result;
}

// 求和运算
VarPtr sum(VarPtr a) {
    double sum_val = 0.0;
    for (double val : a->data()) {
        sum_val += val;
    }
    
    bool requires_grad = a->requires_grad();
    auto result = make_var(sum_val, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                // 梯度广播到所有元素
                std::vector<double> grad_a(a->size(), grad_output[0]);
                a->backward(grad_a);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
    }
    
    return result;
}

// 平均值运算
VarPtr mean(VarPtr a) {
    double sum_val = 0.0;
    for (double val : a->data()) {
        sum_val += val;
    }
    double mean_val = sum_val / a->size();
    
    bool requires_grad = a->requires_grad();
    auto result = make_var(mean_val, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                // 梯度平均分配到所有元素
                double grad_per_element = grad_output[0] / a->size();
                std::vector<double> grad_a(a->size(), grad_per_element);
                a->backward(grad_a);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
    }
    
    return result;
}

/**
 * 激活函数
 */

// ReLU激活函数
VarPtr relu(VarPtr a) {
    std::vector<double> result_data(a->size());
    
    for (size_t i = 0; i < a->size(); ++i) {
        result_data[i] = std::max(0.0, a->data()[i]);
    }
    
    bool requires_grad = a->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size());
                for (size_t i = 0; i < a->size(); ++i) {
                    // ReLU的导数：x > 0 时为1，否则为0
                    grad_a[i] = (a->data()[i] > 0) ? grad_output[i] : 0.0;
                }
                a->backward(grad_a);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
    }
    
    return result;
}

// Sigmoid激活函数
VarPtr sigmoid(VarPtr a) {
    std::vector<double> result_data(a->size());
    
    for (size_t i = 0; i < a->size(); ++i) {
        result_data[i] = 1.0 / (1.0 + std::exp(-a->data()[i]));
    }
    
    bool requires_grad = a->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, result](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size());
                for (size_t i = 0; i < a->size(); ++i) {
                    // Sigmoid的导数：sigmoid(x) * (1 - sigmoid(x))
                    double sigmoid_val = result->data()[i];
                    grad_a[i] = grad_output[i] * sigmoid_val * (1.0 - sigmoid_val);
                }
                a->backward(grad_a);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
    }
    
    return result;
}

// Tanh激活函数
VarPtr tanh_activation(VarPtr a) {
    std::vector<double> result_data(a->size());
    
    for (size_t i = 0; i < a->size(); ++i) {
        result_data[i] = std::tanh(a->data()[i]);
    }
    
    bool requires_grad = a->requires_grad();
    auto result = make_var(result_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [a, result](const std::vector<double>& grad_output) {
            if (a->requires_grad()) {
                std::vector<double> grad_a(a->size());
                for (size_t i = 0; i < a->size(); ++i) {
                    // Tanh的导数：1 - tanh²(x)
                    double tanh_val = result->data()[i];
                    grad_a[i] = grad_output[i] * (1.0 - tanh_val * tanh_val);
                }
                a->backward(grad_a);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(a);
    }
    
    return result;
}

/**
 * 损失函数
 */

// 均方误差损失函数
VarPtr mse_loss(VarPtr predictions, VarPtr targets) {
    if (predictions->size() != targets->size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    // 计算 MSE = mean((predictions - targets)^2)
    auto diff = sub(predictions, targets);
    auto squared_diff = mul(diff, diff);
    return mean(squared_diff);
}

// 二元交叉熵损失函数
VarPtr binary_cross_entropy_loss(VarPtr predictions, VarPtr targets) {
    if (predictions->size() != targets->size()) {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    
    std::vector<double> loss_data(predictions->size());
    
    for (size_t i = 0; i < predictions->size(); ++i) {
        double pred = predictions->data()[i];
        double target = targets->data()[i];
        
        // 防止log(0)
        pred = std::max(1e-15, std::min(1.0 - 1e-15, pred));
        
        // BCE = -[target * log(pred) + (1 - target) * log(1 - pred)]
        loss_data[i] = -(target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred));
    }
    
    bool requires_grad = predictions->requires_grad() || targets->requires_grad();
    auto result = make_var(loss_data, requires_grad);
    
    if (requires_grad) {
        auto grad_fn = [predictions, targets](const std::vector<double>& grad_output) {
            if (predictions->requires_grad()) {
                std::vector<double> grad_pred(predictions->size());
                for (size_t i = 0; i < predictions->size(); ++i) {
                    double pred = predictions->data()[i];
                    double target = targets->data()[i];
                    
                    // 防止除零
                    pred = std::max(1e-15, std::min(1.0 - 1e-15, pred));
                    
                    // BCE对prediction的导数：-(target/pred - (1-target)/(1-pred))
                    grad_pred[i] = grad_output[i] * (-(target / pred) + (1.0 - target) / (1.0 - pred));
                }
                predictions->backward(grad_pred);
            }
        };
        
        result->set_grad_fn(grad_fn);
        result->add_child(predictions);
        result->add_child(targets);
    }
    
    return mean(result);  // 返回平均损失
}
VarPtr operator+(VarPtr a, VarPtr b) { return add(a, b); }
VarPtr operator-(VarPtr a, VarPtr b) { return sub(a, b); }
VarPtr operator*(VarPtr a, VarPtr b) { return mul(a, b); }
VarPtr operator/(VarPtr a, VarPtr b) { return div(a, b); }
VarPtr operator^(VarPtr a, double exponent) { return pow(a, exponent); }
