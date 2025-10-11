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
#include <ranges>

// 前向声明
class Variable;
using VarPtr = std::shared_ptr<Variable>;

/**
 * 支持自动微分的变量类
 */
enum Nodetype {
    intermediate,
    parameter,
    input
};
struct Edge {
    VarPtr parent;
    VarPtr child;
    bool updated;
    Edge(VarPtr parent, VarPtr child, bool updated) : parent(parent), child(child), updated(updated) {}
};
std::ostream& operator<<(std::ostream &os, const Nodetype &type){
    switch(type){
        case intermediate:
            os << "intermediate";
            break;
        case parameter:
            os << "parameter";
            break;
        case input:
            os << "input";
            break;
        default:
            os << "unknown";
    }
    return os;
}
class Variable
{
private:
    mutable std::vector<double> accumulated_grad_;
    mutable bool updated_ = false;

    std::vector<double> data_;
    std::vector<double> grad_;
    // bool requires_grad_;
    Nodetype type_;
    std::function<void(const std::vector<double> &)> grad_fn_;
    std::function<void()> forward_fn_; 
    std::vector<Edge*> children_;
    std::vector<Edge*> parents_;
    std::vector<size_t> shape_;
public:
    bool has_grad() const { return type_ == parameter || type_ == intermediate; }
    /**
     * 构造函数
     * @param data 数值数据
     * @param type 节点类型
     * @param shape 张量形状（可选）
     */
    explicit Variable(double value, Nodetype type, const std::vector<size_t> &shape = {})
        : data_({value}), type_(type), shape_(shape)
    {
        if (type_ == parameter || type_ == intermediate)
        {
            grad_ = std::vector<double>(1, 0.0);
            accumulated_grad_ = std::vector<double>(1, 0.0);
        }
        if (shape_.empty())
        {
            shape_ = {data_.size()};
        }
        check_validity();
    }

    /**
     * 向量构造函数
     * @param data 数值向量
     * @param type 节点类型
     * @param shape 张量形状（可选）
     */
    explicit Variable(const std::vector<double> &data, Nodetype type, const std::vector<size_t> &shape = {})
        : data_(data), type_(type), shape_(shape)
    {
        if (has_grad())
        {
            grad_ = std::vector<double>(data.size(), 0.0);
            accumulated_grad_ = std::vector<double>(data.size(), 0.0);
        }
        if (shape_.empty())
        {
            shape_ = {data_.size()};
        }
        check_validity();
    }
    void check_validity()
    {
        size_t total = 1;
        for (size_t idx = 0; idx < ndim(); ++idx)
        {
            size_t dim = shape_[idx];
            if (dim <= 0)
            {
                throw std::runtime_error("Invalid shape dimension" + std::to_string(idx) + ": " + std::to_string(dim));
            }
            total *= dim;
        }
        if (total != data_.size())
        {
            throw std::runtime_error("Data size does not match shape");
        }
    }
    // check if it is some simple shapes like scalar, vector and matrix
    int ndim() const
    {
        return shape_.size();
    }
    bool is_scalar() const
    {
        return size() == 1;
    }
    bool is_vector() const
    {
        return ndim() == 1 && shape_[0] > 1;
    }
    bool is_matrix() const
    {
        return ndim() == 2 && shape_[0] > 1 && shape_[1] > 1;
    }
    // 访问器
    const std::vector<double> &data() const { return data_; }
    const std::vector<double> &grad() const { return grad_; }
    inline std::vector<Edge*>& children() { return children_; }
    inline std::vector<Edge*>& parents() { return parents_; }
    Nodetype type() const { return type_; }
    size_t size() const { return data_.size(); }
    std::vector<size_t> shape() const { return shape_; }

    // update the data using gradient and learning rate
    void update(double learning_rate)
    {
        for (size_t i = 0; i < data_.size(); i++)
        {
            data_[i] -= learning_rate * grad_[i];
        }
    }
    // 获取标量值（仅适用于单元素）
    double item() const
    {
        if (!is_scalar())
        {
            throw std::runtime_error("item() can only be called on scalar variables");
        }
        return data_[0];
    }

    // 获取梯度标量值（仅适用于单元素）
    double grad_item() const
    {
        if (!has_grad() || !is_scalar())
        {
            throw std::runtime_error("grad_item() can only be called on scalar variables with gradients");
        }
        return grad_[0];
    }
    void set_input(const double new_data){
        if(type_ != input){
            throw std::runtime_error("Only input variable can set data");
        }
        if(data_.size() != 1){
            throw std::runtime_error("This variable is not a scalar");
        }
        data_[0] = new_data;
    }
    void set_input(const std::vector<double>& new_data){
        if(type_ != input){
            throw std::runtime_error("Only input variable can set data");
        }
        if(new_data.size() != data_.size()){
            throw std::runtime_error("New data size does not match");
        }
        data_ = new_data;
    }
    /**
     * 设置梯度函数
     */
    void set_grad_fn(std::function<void(const std::vector<double> &)> grad_fn)
    {
        grad_fn_ = grad_fn;
    }
    void set_forward_fn(std::function<void()> forward_fn)
    {
        forward_fn_ = forward_fn;
    }
    /**
     * 添加子节点
     */
    void add_child(Edge* edge)
    {
        children_.push_back(edge);
    }
    
    void add_parent(Edge* edge)
    {
        parents_.push_back(edge);
    }
    /**
     * 前向计算
     */
    void calc(){
        if(type_ != intermediate){
            throw std::runtime_error("Only intermediate nodes can be calculated");
        }
        for(auto & edge: children_){
            edge->child->calc();
        }
        if(this->forward_fn_)
            this->forward_fn_();
    }
    bool updated() const { return updated_; }
    // 累积梯度（不立即传播）
    void accumulate_gradient(const std::vector<double>& grad_input) {
        if (!has_grad()) return;

        // 累积梯度
        for (size_t i = 0; i < accumulated_grad_.size() && i < grad_input.size(); ++i) {
            accumulated_grad_[i] += grad_input[i];
        }

        // 检查是否所有父节点都已发送梯度
        bool all_gradients_received = true;
        for (const auto& edge : parents()) {
            if (edge->parent && edge->parent->has_grad()) {
                // 如果父节点需要梯度，检查是否已经从该父节点接收到梯度
                if (edge->updated == false) {
                    all_gradients_received = false;
                    break;
                }
            }
        }

        // 如果所有梯度都已接收，执行反向传播
        if (all_gradients_received) {
            // 将累积的梯度复制到正式的梯度中
            grad_ = accumulated_grad_;
            updated_ = true;
            // 现在可以安全地向子节点传播梯度
            if (grad_fn_) {
                grad_fn_(grad_);
            }
        }
    }
    /**
     * 反向传播计算梯度
     * @param grad_output 从上游传来的梯度
     */
    void backward(const std::vector<double> &grad_output = {})
    {
        if (!has_grad())
            return;

        // 如果是标量输出且没有指定梯度，设为1 (优化目标)
        if (grad_output.empty() && parents_.empty())
        {
            if(data_.size() != 1){
                throw std::runtime_error("Gradient can only be computed for scalar outputs");
            }
            accumulated_grad_[0] = 1.0;
            grad_ = accumulated_grad_;
            updated_ = true;
            // 如果有梯度函数，继续反向传播
            if (grad_fn_)
            {
                grad_fn_(grad_);
            }
        }else {
            throw std::runtime_error("Cannot backward() non-root nodes. Use accumulate_gradient() instead.");
        }

        
    }

    /**
     * 清零梯度
     */
    void zero_grad()
    {
        if (has_grad())
        {
            std::fill(grad_.begin(), grad_.end(), 0.0);
            std::fill(accumulated_grad_.begin(), accumulated_grad_.end(), 0.0);
            updated_ = false;
        }
    }
    // void forward(){
    //     if(type_ == intermediate && forward_fn_){
    //         forward_fn_();
    //     }
    // }
    VarPtr flatten(){
        this->shape_ = {this->size()};
        return VarPtr(this);
    }
    void zero_grad_recursive()
    {
        zero_grad();
        for (auto &edge : children_)
        {
            edge->updated = false;
            if (edge->child != nullptr)
            edge->child->zero_grad_recursive();
        }
    }
    /**
     * 打印变量信息
     */
    void print() const
    {
        std::cout << "Variable(data=[";
        for (size_t i = 0; i < data_.size(); ++i)
        {
            std::cout << std::fixed << std::setprecision(4) << data_[i];
            if (i < data_.size() - 1)
                std::cout << ", ";
        }
        std::cout << "], shape=(";
        for(size_t i = 0; i < ndim(); ++i)
        {
            std::cout << shape_[i];
            if (i < ndim() - 1)
                std::cout << ", ";
        }
        std::cout << "), type=" << type_;

        if (has_grad() && !grad_.empty())
        {
            std::cout << ", grad=[";
            for (size_t i = 0; i < grad_.size(); ++i)
            {
                std::cout << std::fixed << std::setprecision(4) << grad_[i];
                if (i < grad_.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << ")" << std::endl;
    }
    // 通过扁平索引获取多维索引
    std::vector<int> PlainItemIndex(size_t flat_index)
    {
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        std::vector<int> idx(ndim());
        size_t temp = flat_index;
        for (int i = ndim() - 1; i >= 0; i--)
        {
            idx[i] = temp % shape_[i];
            temp /= shape_[i];
        }
        return idx;
    }
    // 索引访问函数，支持多维索引
    size_t ItemIndex(const std::vector<int> &idx)
    {
        if (idx.size() != shape().size())
        { // dimension match
            throw std::runtime_error("Index dimension does not match variable dimension");
        }
        for (size_t i = 0; i < idx.size(); i++)
        {
            if (idx[i] < 0 || static_cast<size_t>(idx[i]) >= shape()[i])
            {
                throw std::runtime_error("Index out of bounds at dimension " + std::to_string(i));
            }
        }
        size_t flat_index = 0;
        for (size_t i = 0; i < idx.size(); i++)
        {
            size_t stride = 1;
            for (size_t j = i + 1; j < shape().size(); j++)
            {
                stride *= shape()[j];
            }
            flat_index += idx[i] * stride;
        }
        return flat_index;
    }
    // size_t ItemIndex(const std::vector<size_t> &idx)
    // {
    //     if (idx.size() != shape().size())
    //     { // dimension match
    //         throw std::runtime_error("Index dimension does not match variable dimension");
    //     }
    //     return ItemIndex(std::vector<int>(idx.begin(), idx.end()));
    // }

    // 通过多维索引获取元素
    double& Item(const std::vector<int> &idx)
    {
        return data_[ItemIndex(idx)];
    }
    double& Item(size_t flat_index)
    {
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        return data_[flat_index];
    }
    // 通过多维索引获取梯度
    double GradItem(const std::vector<int> &idx)
    {
        if (!has_grad())
        {
            throw std::runtime_error("This variable does not require gradient");
        }
        return grad()[ItemIndex(idx)];
    }
    double& GradItem(size_t flat_index)
    {
        if (!has_grad())
        {
            throw std::runtime_error("This variable does not require gradient");
        }
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        return grad_[flat_index];
    }

};

// 工具函数：创建Variable的智能指针


VarPtr make_var(double value)
{
    return std::make_shared<Variable>(value, intermediate);
}

VarPtr make_var(const std::vector<double> &data, const std::vector<size_t> &shape = {})
{
    return std::make_shared<Variable>(data, intermediate, shape);
}

VarPtr make_param(double value, const std::vector<size_t> &shape = {})
{
    return std::make_shared<Variable>(value, parameter, shape);
}
VarPtr make_param(const std::vector<double> &data, const std::vector<size_t> &shape = {})
{
    return std::make_shared<Variable>(data, parameter, shape);
}
VarPtr make_input(double value, const std::vector<size_t> &shape = {})
{
    return std::make_shared<Variable>(value, input, shape);
}
VarPtr make_input(const std::vector<double> &data, const std::vector<size_t> &shape = {})
{
    return std::make_shared<Variable>(data, input, shape);
}

#include "operations.hpp"

VarPtr operator+(VarPtr a, VarPtr b) { return add(a, b); }
VarPtr operator-(VarPtr a, VarPtr b) { return sub(a, b); }
VarPtr operator*(VarPtr a, VarPtr b) { return mul(a, b); }
// VarPtr operator/(VarPtr a, VarPtr b) { return div(a, b); }  // div function is commented out
VarPtr operator^(VarPtr a, double exponent) { return pow_elementwise(a, exponent); }
