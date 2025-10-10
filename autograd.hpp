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
using namespace std;
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
    std::vector<double> data_;
    std::vector<double> grad_;
    // bool requires_grad_;
    Nodetype type_;
    std::function<void(const std::vector<double> &)> grad_fn_;
    std::function<void()> forward_fn_; 
    std::vector<VarPtr> children_;
    std::vector<size_t> shape_;
public:
    bool has_grad() const { return type_ == parameter || type_ == intermediate; }
    /**
     * 构造函数
     * @param data 数值数据
     * @param requires_grad 是否需要梯度计算
     */
    explicit Variable(double value, Nodetype type, const std::vector<size_t> &shape = {})
        : data_({value}), type_(type), shape_(shape)
    {
        if (type_ == parameter || type_ == intermediate)
        {
            grad_ = std::vector<double>(1, 0.0);
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
     * @param requires_grad 是否需要梯度计算
     */
    explicit Variable(const std::vector<double> &data, Nodetype type, const std::vector<size_t> &shape = {})
        : data_(data), type_(type), shape_(shape)
    {
        if (has_grad())
        {
            grad_ = std::vector<double>(data.size(), 0.0);
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
        for (size_t idx = 0; idx < shape_.size(); ++idx)
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
    void set_data(std::vector<double>& new_data){
        // if(!is_variable()){
        //     throw std::runtime_error("Cannot set data for constant variable");
        // }
        if(new_data.size() != data_.size()){
            throw std::runtime_error("New data size does not match");
        }
        data_ = new_data;
    }
    // 获取标量值（仅适用于单元素）
    double& item()
    {
        if (!is_scalar())
        {
            throw std::runtime_error("item() can only be called on scalar variables");
        }
        return data_[0];
    }

    // 获取梯度标量值（仅适用于单元素）
    double& grad_item()
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
    void add_child(VarPtr child)
    {
        children_.push_back(child);
    }
    void calc(){
        for(auto & child: children_){
            child->calc();
        }
        this->forward();
    }
    /**
     * 反向传播计算梯度
     * @param grad_output 从上游传来的梯度
     */
    void backward(const std::vector<double> &grad_output = {})
    {
        if (!has_grad())
            return;

        std::vector<double> grad_out = grad_output;

        // 如果是标量输出且没有指定梯度，设为1 (优化目标)
        if (grad_out.empty())
        {
            grad_out = std::vector<double>(data_.size(), 1.0);
        }

        // 累积梯度
        for (size_t i = 0; i < grad_.size(); ++i)
        {
            grad_[i] += grad_out[i];
        }

        // 如果有梯度函数，继续反向传播
        if (grad_fn_)
        {
            grad_fn_(grad_out);
        }
    }

    void forward(){
        if(is_variable() && forward_fn_){
            forward_fn_();
        }
    }
    void recursive_zero_grad(){
        zero_grad();
        for(auto &child: children_){
            child->recursive_zero_grad();
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
        }
    }
    void forward(){
        if(type_ == intermediate && forward_fn_){
            forward_fn_();
        }
    }
    void zero_grad_recursive()
    {
        zero_grad();
        for (auto &child : children_)
        {
            child->zero_grad_recursive();
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
    const std::vector<int> PlainItemIndex(const size_t flat_index) const
    {
        if (flat_index >= size())
        {
            throw std::runtime_error("Flat index out of bounds");
        }
        std::vector<int> idx(ndim(), 0);
        size_t temp = flat_index;
        for (int i = ndim() - 1; i >= 0; i--)
        {
            idx[i] = temp % shape_[i];
            temp /= shape_[i];
        }
        return idx;
    }
    // 索引访问函数，支持多维索引
    const size_t ItemIndex(const std::vector<int> &idx) const
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

    // 通过多维索引获取元素
    double Item(const std::vector<int> &idx)
    {
        return data()[ItemIndex(idx)];
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

// get broadcasted index
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

/**
 * 基础数学运算函数
 */

// 加法运算
VarPtr add(VarPtr a, VarPtr b)
{

    size_t len = std::max(a->ndim(), b->ndim());
    std::vector<size_t> result_shape(len);
    for (size_t i=0; i < len; i++)
    { // 确定结果shape
        size_t a_dim = (i <  a->ndim()) ? a->shape()[a->ndim()-i-1] : 1;
        size_t b_dim = (i < b->ndim()) ? b->shape()[b->ndim()-i-1] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
        {
            throw std::runtime_error("Incompatible sizes for addition");
        }
        result_shape[len-1-i] = std::max(a_dim, b_dim);
    }
    size_t result_size = 1; // 计算结果总大小
    for (size_t dim : result_shape)
    {
        result_size *= dim;
    }

    std::vector<double> result_data(result_size);
    auto result = make_var(result_data, result_shape);
    // 执行加法（支持广播）
    auto forward_fn = [a, b, result, result_size, result_shape]() {
        // 先递归计算依赖的节点
        if (a->type() == intermediate) a->forward();
        if (b->type() == intermediate) b->forward();
        
        for (size_t i = 0; i < result_size; i++)
        {
            // 直接使用 result 变量计算索引，避免创建临时变量
            std::vector<int> result_idx = result->PlainItemIndex(i);
            
            // 计算广播后的 a 和 b 索引
            std::vector<int> a_idx = get_broadcast_idx(result_idx, a->shape());
            std::vector<int> b_idx = get_broadcast_idx(result_idx, b->shape());
            
            double a_val = (a->shape().empty()) ? a->data()[0] : a->Item(a_idx);
            double b_val = (b->shape().empty()) ? b->data()[0] : b->Item(b_idx);
            result->Item(i) = a_val + b_val;
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    


    auto grad_fn = [a, b, result, result_shape, result_size](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size(), 0.0);
            for (size_t i = 0; i < result_size; i++)
            {
                std::vector<int> a_idx = get_broadcast_idx(result->PlainItemIndex(i), a->shape());
                grad_a[a->ItemIndex(a_idx)] += grad_output[i];
            }
            a->backward(grad_a);
        }
        if (b->has_grad())
        {
            std::vector<double> grad_b(b->size(), 0.0);
            for (size_t i = 0; i < result_size; i++)
            {
                std::vector<int> b_idx = get_broadcast_idx(result->PlainItemIndex(i), b->shape());
                grad_b[b->ItemIndex(b_idx)] += grad_output[i];
            }
            b->backward(grad_b);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);
    result->add_child(b);

    return result;
}

// 减法运算
VarPtr sub(VarPtr a, VarPtr b){

    size_t len = std::max(a->ndim(), b->ndim());
    std::vector<size_t> result_shape(len);
    for (size_t i=0; i < len; i++)
    { // 确定结果shape

        size_t a_dim = (i <  a->ndim()) ? a->shape()[a->ndim()-i-1] : 1;
        size_t b_dim = (i < b->ndim()) ? b->shape()[b->ndim()-i-1] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1)
        {
            throw std::runtime_error("Incompatible sizes for subtraction");
        }
        result_shape[len-1-i] = std::max(a_dim, b_dim);
    }
    size_t result_size = 1; // 计算结果总大小
    for (size_t dim : result_shape)
    {
        result_size *= dim;
    }

    std::vector<double> result_data(result_size);
    auto result = make_var(result_data, result_shape);
    
    // 添加前向函数
    auto forward_fn = [a, b, result, result_size, result_shape]() {
        for (size_t i = 0; i < result_size; i++)
        {
            // 直接使用 result 变量计算索引，避免创建临时变量
            std::vector<int> result_idx = result->PlainItemIndex(i);
            
            // 计算广播后的 a 和 b 索引
            std::vector<int> a_idx = get_broadcast_idx(result_idx, a->shape());
            std::vector<int> b_idx = get_broadcast_idx(result_idx, b->shape());
            
            double a_val = (a->shape().empty()) ? a->data()[0] : a->Item(a_idx);
            double b_val = (b->shape().empty()) ? b->data()[0] : b->Item(b_idx);
            result->Item(i) = a_val - b_val;
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    if (a->has_grad() || b->has_grad())
    {
        auto grad_fn = [a, b, result, result_shape, result_size](const std::vector<double> &grad_output)
        {
            if (a->has_grad())
            {
                std::vector<double> grad_a(a->size(), 0.0);
                for (size_t i = 0; i < result_size; i++)
                {
                    std::vector<int> a_idx = get_broadcast_idx(result->PlainItemIndex(i), a->shape());
                    grad_a[a->ItemIndex(a_idx)] += grad_output[i];
                }
                a->backward(grad_a);
            }
            if (b->has_grad())
            {
                std::vector<double> grad_b(b->size(), 0.0);
                for (size_t i = 0; i < result_size; i++)
                {
                    std::vector<int> b_idx = get_broadcast_idx(result->PlainItemIndex(i), b->shape());
                    grad_b[b->ItemIndex(b_idx)] -= grad_output[i];
                }
                b->backward(grad_b);
            }
        };

        result->set_grad_fn(grad_fn);
    }
    if(is_var){
        result->set_forward_fn(forward_fn);
    }else {
        forward_fn();
    }
    result->add_child(a);
    result->add_child(b);

    return result;
}
// 乘法运算 - 支持张量乘法


// 元素级乘法
VarPtr mul_elementwise(VarPtr a, VarPtr b) {
    size_t len = std::max(a->ndim(), b->ndim());
    std::vector<size_t> result_shape(len);
    
    // 计算结果形状（从右对齐的广播）
    for (size_t i = 0; i < len; i++) {
        size_t a_dim = (i < a->ndim()) ? a->shape()[a->ndim()-i-1] : 1;
        size_t b_dim = (i < b->ndim()) ? b->shape()[b->ndim()-i-1] : 1;
        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("Incompatible sizes for element-wise multiplication");
        }
        result_shape[len-1-i] = std::max(a_dim, b_dim);
    }
    
    size_t result_size = 1;
    for (size_t dim : result_shape) {
        result_size *= dim;
    }
    
    std::vector<double> result_data(result_size);
    auto result = make_var(result_data, result_shape);
    
    // 添加前向函数
    auto forward_fn = [a, b, result, result_size, result_shape]() {
        for (size_t i = 0; i < result_size; i++) {
            // 直接使用 result 变量计算索引，避免创建临时变量
            std::vector<int> result_idx = result->PlainItemIndex(i);
            
            std::vector<int> a_idx = get_broadcast_idx(result_idx, a->shape());
            std::vector<int> b_idx = get_broadcast_idx(result_idx, b->shape());
            
            double a_val = (a->shape().empty()) ? a->data()[0] : a->Item(a_idx);
            double b_val = (b->shape().empty()) ? b->data()[0] : b->Item(b_idx);
            result->Item(i) = a_val * b_val;
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    
    auto grad_fn = [a, b, result, result_shape, result_size](const std::vector<double> &grad_output) {
        if (a->has_grad()) {
            std::vector<double> grad_a(a->size(), 0.0);
            for (size_t i = 0; i < result_size; i++) {
                std::vector<int> a_idx = get_broadcast_idx(result->PlainItemIndex(i), a->shape());
                std::vector<int> b_idx = get_broadcast_idx(result->PlainItemIndex(i), b->shape());
                
                double b_val = (b->shape().empty()) ? b->data()[0] : b->Item(b_idx);
                
                if (a->shape().empty()) {
                    grad_a[0] += grad_output[i] * b_val;
                } else {
                    grad_a[a->ItemIndex(a_idx)] += grad_output[i] * b_val;
                }
            }
            a->backward(grad_a);
        }
        
        if (b->has_grad()) {
            std::vector<double> grad_b(b->size(), 0.0);
            for (size_t i = 0; i < result_size; i++) {
                std::vector<int> a_idx = get_broadcast_idx(result->PlainItemIndex(i), a->shape());
                std::vector<int> b_idx = get_broadcast_idx(result->PlainItemIndex(i), b->shape());
                
                double a_val = (a->shape().empty()) ? a->data()[0] : a->Item(a_idx);
                
                if (b->shape().empty()) {
                    grad_b[0] += grad_output[i] * a_val;
                } else {
                    grad_b[b->ItemIndex(b_idx)] += grad_output[i] * a_val;
                }
            }
            b->backward(grad_b);
        }
    };
    result->set_grad_fn(grad_fn);
    result->add_child(a);
    result->add_child(b);

    
    return result;

}

VarPtr tensor(VarPtr a, VarPtr b){
    size_t len = a->size() * b->size();
    bool requires_grad = a->requires_grad() || b->requires_grad();
    std::vector<size_t> shape(a->ndim() + b->ndim());
    // fill in the shape
    for(size_t i=0; i < a->ndim(); i++){
        shape[i] = a->shape()[i];
    }
    for(size_t j=0; j < b->ndim(); j++){
        shape[a->ndim()+j] = b->shape()[j];
    }
    bool is_var = a->is_variable() || b->is_variable();
    VarPtr result = make_var(std::vector<double>(len), requires_grad, shape,is_var);
    if(result->requires_grad()){
        auto grad_fn = [a, b, result, len](const std::vector<double> &grad_output){
            if(a->requires_grad()){
                std::vector<double> grad_a(a->size(), 0.0);
                for(size_t i = 0; i < len; i++){
                    std::vector<int> result_idx = result->PlainItemIndex(i);
                    std::vector<int> a_idx(result_idx.begin(), result_idx.begin()+a->ndim());
                    std::vector<int> b_idx(result_idx.begin()+a->ndim(), result_idx.end());
                    grad_a[a->ItemIndex(a_idx)] += grad_output[i] * b->Item(b_idx);
                }
                a->backward(grad_a);
            }
            if(b->requires_grad()){
                std::vector<double> grad_b(b->size(), 0.0);
                for(size_t i = 0; i < len; i++){
                    std::vector<int> result_idx = result->PlainItemIndex(i);
                    std::vector<int> a_idx(result_idx.begin(), result_idx.begin()+a->ndim());
                    std::vector<int> b_idx(result_idx.begin()+a->ndim(), result_idx.end());
                    grad_b[b->ItemIndex(b_idx)] += grad_output[i] * a->Item(a_idx);
                }
                b->backward(grad_b);
            }
        };
        result->set_grad_fn(grad_fn);
    }
    auto forward_fn = [a, b, result, len](){
        for(size_t i = 0; i < len; i++){
            std::vector<int> result_idx = result->PlainItemIndex(i);
            result->Item(i) = a->Item(std::vector<int>(result_idx.begin(), result_idx.begin()+a->ndim())) *
                                b->Item(std::vector<int>(result_idx.begin()+a->ndim(), result_idx.end()));
        }
    };
    if(is_var){
        result->set_forward_fn(forward_fn);
    }else{
        forward_fn();
    }
    result->add_child(a);
    result->add_child(b);
    return result;
}
VarPtr mul(VarPtr a, VarPtr b, int axis_a = -1, int axis_b = -1)
{
    if (axis_a == -1 && axis_b == -1) {
        return mul_elementwise(a, b);
    }
    // 张量乘法：沿指定轴收缩
    // 检查轴的有效性
    if (axis_a < 0) axis_a += a->ndim();
    if (axis_b < 0) axis_b += b->ndim();
    
    if (axis_a >= static_cast<int>(a->ndim()) || axis_b >= static_cast<int>(b->ndim()) ||
        axis_a < 0 || axis_b < 0) {
        throw std::runtime_error("Invalid axis for tensor multiplication");
    }
    
    size_t contract_dim_a = a->shape()[axis_a];
    size_t contract_dim_b = b->shape()[axis_b];
    
    if (contract_dim_a != contract_dim_b) {
        throw std::runtime_error("Incompatible dimensions for tensor contraction");
    }
    
    // 计算结果形状
    std::vector<size_t> result_shape;
    for (size_t i = 0; i < a->ndim(); i++) {
        if (i != static_cast<size_t>(axis_a)) {
            result_shape.push_back(a->shape()[i]);
        }
    }
    for (size_t i = 0; i < b->ndim(); i++) {
        if (i != static_cast<size_t>(axis_b)) {
            result_shape.push_back(b->shape()[i]);
        }
    }
    
    if (result_shape.empty()) {
        result_shape = {1}; // 标量结果
    }
    
    size_t result_size = 1;
    for (size_t dim : result_shape) {
        result_size *= dim;
    }
    
    std::vector<double> result_data(result_size, 0.0);
    auto result = make_var(result_data, result_shape);
    
    // 添加前向函数
    auto forward_fn = [a, b, result, result_size, result_shape, axis_a, axis_b, contract_dim_a]() {
        // 执行张量乘法
        for (size_t res_i = 0; res_i < result_size; res_i++) {
            std::vector<int> result_idx = result->PlainItemIndex(res_i);
            
            // 构造 a 和 b 的完整索引
            std::vector<int> full_a_idx(a->shape().size());
            std::vector<int> full_b_idx(b->shape().size());
            
            // 填充 a 的索引（跳过收缩轴）
            size_t a_idx_pos = 0;
            for (size_t i = 0; i < a->shape().size(); i++) {
                if (i != static_cast<size_t>(axis_a)) {
                    if (a_idx_pos < result_idx.size()) {
                        full_a_idx[i] = result_idx[a_idx_pos++];
                    }
                }
            }
            
            // 填充 b 的索引（跳过收缩轴）
            size_t b_idx_pos = a->shape().size() - 1; // b 的索引从 a 的维度之后开始
            if (static_cast<size_t>(axis_a) < a->shape().size()) {
                b_idx_pos--; // 因为跳过了 a 的收缩轴
            }
            
            for (size_t i = 0; i < b->shape().size(); i++) {
                if (i != static_cast<size_t>(axis_b)) {
                    if (b_idx_pos < result_idx.size()) {
                        full_b_idx[i] = result_idx[b_idx_pos++];
                    }
                }
            }
            
            // 初始化结果为0
            result->Item(res_i) = 0.0;
            // 沿收缩维度求和
            for (size_t k = 0; k < contract_dim_a; k++) {
                full_a_idx[axis_a] = k;
                full_b_idx[axis_b] = k;
                
                double a_val = a->Item(full_a_idx);
                double b_val = b->Item(full_b_idx);
                result->Item(res_i) += a_val * b_val;
            }
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [a, b, result, axis_a, axis_b, contract_dim_a](const std::vector<double> &grad_output) {
        if (a->has_grad()) {
            std::vector<double> grad_a(a->size(), 0.0);
            // 实现张量乘法的反向传播
            // grad_a[i,j,k] = sum_l (grad_output[i,j,l] * b[k,l])  (假设沿最后一维收缩)
            // 这里需要根据具体的轴来计算
            for (size_t a_i = 0; a_i < a->size(); a_i++) {
                std::vector<int> a_idx = a->PlainItemIndex(a_i);
                
                for (size_t res_i = 0; res_i < result->size(); res_i++) {
                    std::vector<int> result_idx = result->PlainItemIndex(res_i);
                    
                    // 构造对应的 b 索引
                    std::vector<int> b_idx(b->shape().size());
                    b_idx[axis_b] = a_idx[axis_a];
                    
                    // 填充 b 的其他维度索引
                    size_t b_dim_pos = a->shape().size() - 1;
                    if (static_cast<size_t>(axis_a) < a->shape().size()) b_dim_pos--;
                    
                    for (size_t i = 0; i < b->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_b)) {
                            if (b_dim_pos < result_idx.size()) {
                                b_idx[i] = result_idx[b_dim_pos++];
                            }
                        }
                    }
                    
                    // 检查 a 的索引是否匹配结果索引
                    bool match = true;
                    size_t a_dim_pos = 0;
                    for (size_t i = 0; i < a->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_a)) {
                            if (a_dim_pos < result_idx.size() && a_idx[i] != result_idx[a_dim_pos]) {
                                match = false;
                                break;
                            }
                            a_dim_pos++;
                        }
                    }
                    
                    if (match) {
                        double b_val = b->Item(b_idx);
                        grad_a[a_i] += grad_output[res_i] * b_val;
                    }
                }
            }
            a->backward(grad_a);
        }
        
        if (b->has_grad()) {
            std::vector<double> grad_b(b->size(), 0.0);
            // 类似地计算 b 的梯度
            for (size_t b_i = 0; b_i < b->size(); b_i++) {
                std::vector<int> b_idx = b->PlainItemIndex(b_i);
                
                for (size_t res_i = 0; res_i < result->size(); res_i++) {
                    std::vector<int> result_idx = result->PlainItemIndex(res_i);
                    
                    // 构造对应的 a 索引
                    std::vector<int> a_idx(a->shape().size());
                    a_idx[axis_a] = b_idx[axis_b];
                    
                    // 填充 a 的其他维度索引
                    size_t a_dim_pos = 0;
                    for (size_t i = 0; i < a->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_a)) {
                            if (a_dim_pos < result_idx.size()) {
                                a_idx[i] = result_idx[a_dim_pos++];
                            }
                        }
                    }
                    
                    // 检查 b 的索引是否匹配结果索引
                    bool match = true;
                    size_t b_dim_pos = a->shape().size() - 1;
                    if (static_cast<size_t>(axis_a) < a->shape().size()) b_dim_pos--;
                    
                    for (size_t i = 0; i < b->shape().size(); i++) {
                        if (i != static_cast<size_t>(axis_b)) {
                            if (b_dim_pos < result_idx.size() && b_idx[i] != result_idx[b_dim_pos]) {
                                match = false;
                                break;
                            }
                            b_dim_pos++;
                        }
                    }
                    
                    if (match) {
                        double a_val = a->Item(a_idx);
                        grad_b[b_i] += grad_output[res_i] * a_val;
                    }
                }
            }
            b->backward(grad_b);
        }
    };
        
    result->set_grad_fn(grad_fn);
    result->add_child(a);
    result->add_child(b);
    
    return result;
}

VarPtr pow_elementwise(VarPtr a, double exponent){
    
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data);

    // 添加前向函数
    auto forward_fn = [a, result, exponent]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = std::pow(a->data()[i], exponent);
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto grad_fn = [a, exponent](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // d/dx (x^n) = n * x^(n-1)
                grad_a[i] = grad_output[i] * exponent * std::pow(a->data()[i], exponent - 1);
            }
            a->backward(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);

    return result;
}

// 求和运算
VarPtr sum(VarPtr a)
{
    auto result = make_var(0.0);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        double sum_val = 0.0;
        for (double val : a->data())
        {
            sum_val += val;
        }
        result->Item(0) = sum_val;
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);
    auto grad_fn = [a](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            // 梯度广播到所有元素
            std::vector<double> grad_a(a->size(), grad_output[0]);
            a->backward(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);

    return result;
}

// 平均值运算
VarPtr mean(VarPtr a)
{
    auto result = make_var(0.0);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        double sum_val = 0.0;
        for (double val : a->data())
        {
            sum_val += val;
        }
        double mean_val = sum_val / a->size();
        result->Item(0) = mean_val;
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [a](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            // 梯度平均分配到所有元素
            double grad_per_element = grad_output[0] / a->size();
            std::vector<double> grad_a(a->size(), grad_per_element);
            a->backward(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);
    

    return result;
}

/**
 * 激活函数
 */

// ReLU激活函数
VarPtr relu(VarPtr a)
{
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = std::max(0.0, a->data()[i]);
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);


    auto grad_fn = [a](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // ReLU的导数：x > 0 时为1，否则为0
                grad_a[i] = (a->data()[i] > 0) ? grad_output[i] : 0.0;
            }
            a->backward(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);
    

    return result;
}

// Sigmoid激活函数
VarPtr sigmoid(VarPtr a)
{
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = 1.0 / (1.0 + std::exp(-a->data()[i]));
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [a, result](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // Sigmoid的导数：sigmoid(x) * (1 - sigmoid(x))
                double sigmoid_val = result->data()[i];
                grad_a[i] = grad_output[i] * sigmoid_val * (1.0 - sigmoid_val);
            }
            a->backward(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);
    

    return result;
}

// Tanh激活函数
VarPtr tanh_activation(VarPtr a)
{
    std::vector<double> result_data(a->size());
    auto result = make_var(result_data);

    // 添加前向函数
    auto forward_fn = [a, result]() {
        for (size_t i = 0; i < a->size(); ++i)
        {
            result->Item(i) = std::tanh(a->data()[i]);
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [a, result](const std::vector<double> &grad_output)
    {
        if (a->has_grad())
        {
            std::vector<double> grad_a(a->size());
            for (size_t i = 0; i < a->size(); ++i)
            {
                // Tanh的导数：1 - tanh²(x)
                double tanh_val = result->data()[i];
                grad_a[i] = grad_output[i] * (1.0 - tanh_val * tanh_val);
            }
            a->backward(grad_a);
        }
    };

    result->set_grad_fn(grad_fn);
    result->add_child(a);


    return result;
}

/**
 * 损失函数
 */

// 均方误差损失函数
VarPtr mse_loss(VarPtr predictions, VarPtr targets)
{
    if (predictions->size() != targets->size())
    {
        throw std::runtime_error("Predictions and targets must have the same size");
    }

    // 计算 MSE = mean((predictions - targets)^2)
    auto diff = sub(predictions, targets);
    auto squared_diff = mul(diff, diff);
    return mean(squared_diff);
}

// 二元交叉熵损失函数
VarPtr binary_cross_entropy_loss(VarPtr predictions, VarPtr targets)
{
    if (predictions->size() != targets->size())
    {
        throw std::runtime_error("Predictions and targets must have the same size");
    }
    std::vector<double> loss_data(predictions->size());
    auto result = make_var(loss_data);

    // 添加前向函数
    auto forward_fn = [predictions, targets, result]() {
        for (size_t i = 0; i < predictions->size(); ++i)
        {
            double pred = predictions->data()[i];
            double target = targets->data()[i];
            // 防止log(0)
            pred = std::max(1e-15, std::min(1.0 - 1e-15, pred));
            // BCE = -[target * log(pred) + (1 - target) * log(1 - pred)]
            result->Item(i) = -(target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred));
        }
    };

    // 设置前向函数
    result->set_forward_fn(forward_fn);

    auto grad_fn = [predictions, targets](const std::vector<double> &grad_output)
    {
        if (predictions->has_grad())
        {
            std::vector<double> grad_pred(predictions->size());
            for (size_t i = 0; i < predictions->size(); ++i)
            {
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

    return mean(result); // 返回平均损失
}
VarPtr operator+(VarPtr a, VarPtr b) { return add(a, b); }
VarPtr operator-(VarPtr a, VarPtr b) { return sub(a, b); }
VarPtr operator*(VarPtr a, VarPtr b) { return mul(a, b); }
// VarPtr operator/(VarPtr a, VarPtr b) { return div(a, b); }  // div function is commented out
VarPtr operator^(VarPtr a, double exponent) { return pow_elementwise(a, exponent); }
